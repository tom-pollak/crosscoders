from utils import *
from transformer_lens import ActivationCache
import tqdm
import threading
import time

class Buffer:
    """A dual-buffer system for parallel activation generation and training.

    This class manages two buffers of activations from a pair of models, enabling parallel
    processing where one buffer is being read from while the other is being filled. The system works as follows:

    1. Buffer Structure:
        - Two buffers (buffer_A and buffer_B) alternate between being the 'active_buffer' (being read from)
          and the 'filling_buffer' (being filled with new activations)
        - Each buffer stores activations from both models: shape (buffer_size, 2, d_model)

    2. Threading:
        - Main thread: Handles requests for next batch of activations via next()
        - Generation thread: Continuously fills the inactive buffer in the background

    3. Synchronization:
        - Uses a threading.Event (buffer_ready) to coordinate between threads
        - Event is set when a buffer is ready to be read from
        - Event is cleared when we start filling a new buffer

    4. Operation Flow:
        a. Initialization:
            - Creates both buffers
            - Fills the first active buffer
            - Starts background generation thread

        b. During Training:
            - SAE reads batches from active_buffer via next()
            - When active_buffer is nearly exhausted:
                1. Wait for filling_buffer to be ready
                2. Swap active_buffer and filling_buffer
                3. Start filling the new filling_buffer

    5. Parallel Processing:
        - Uses separate CUDA streams for model A and B to run in parallel
        - Activations are normalized to have average norm sqrt(d_model)

    Args:
        cfg: Configuration dictionary containing:
            - batch_size: Size of batches returned by next()
            - buffer_mult: Buffer size multiplier
            - seq_len: Sequence length for tokens
            - device/device_A/device_B: GPU devices for SAE/model A/model B
            - model_batch_size: Batch size for running models
            - num_tokens: Total number of tokens to process
        model_A: First model to generate activations from
        model_B: Second model to generate activations from
        all_tokens: Tensor of all input tokens to process
    """

    def __init__(self, cfg, model_A, model_B, all_tokens):
        assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)

        # Create two buffers instead of one
        self.buffer_A = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
            device=cfg["device"],
        )
        self.buffer_B = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
            device=cfg["device"],
        )

        self.active_buffer = self.buffer_A  # Buffer currently being read from
        self.filling_buffer = self.buffer_B  # Buffer currently being filled

        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.pointer = 0
        self.normalize = True
        self.all_tokens = all_tokens

        self.estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A, cfg["device_A"])
        self.estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B, cfg["device_B"])
        self.normalisation_factor = torch.tensor(
            [
                self.estimated_norm_scaling_factor_A,
                self.estimated_norm_scaling_factor_B,
            ],
            device=cfg["device"],
        )

        self.stream_A = torch.cuda.Stream(device=cfg["device_A"])
        self.stream_B = torch.cuda.Stream(device=cfg["device_B"])

        # Add event to signal when buffer is ready
        self.buffer_ready = threading.Event()
        self.running = True
        self.generation_thread = None
        self.max_buffer_fills = cfg["num_tokens"] // self.buffer_size
        self.buffer_fills = 0

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, device, n_batches_for_norm_estimate: int = 100):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            _, cache = model.run_with_cache(
                tokens.to(device),
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor
        # Fill first buffer before starting
        print("Filling initial buffer...")
        self.fill_buffer(self.active_buffer)
        self.buffer_ready.set()  # Signal that first buffer is ready

        # Start background thread
        self.start_generation()

    def start_generation(self):
        """Start background activation generation thread"""
        self.generation_thread = threading.Thread(target=self.generation_loop)
        self.generation_thread.start()

    def stop_generation(self):
        """Stop background generation"""
        self.running = False
        if self.generation_thread is not None:
            self.generation_thread.join()

    def fill_buffer(self, buffer):
        """Fill the specified buffer with new activations"""
        fill_pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            for _ in tqdm.trange(0, self.buffer_batches, self.cfg["model_batch_size"]):
                if not self.running:
                    return

                tokens = self.all_tokens[
                    self.token_pointer : min(
                        self.token_pointer + self.cfg["model_batch_size"],
                        self.buffer_batches
                    )
                ]

                with torch.cuda.stream(self.stream_A):
                    _, cache_A = self.model_A.run_with_cache(
                        tokens.to(self.cfg["device_A"]),
                        names_filter=self.cfg["hook_point"]
                    )

                with torch.cuda.stream(self.stream_B):
                    _, cache_B = self.model_B.run_with_cache(
                        tokens.to(self.cfg["device_B"]),
                        names_filter=self.cfg["hook_point"]
                    )

                torch.cuda.synchronize(device=self.cfg["device_A"])
                torch.cuda.synchronize(device=self.cfg["device_B"])

                acts = torch.stack([
                    cache_A[self.cfg["hook_point"]].to(self.cfg["device"]),
                    cache_B[self.cfg["hook_point"]].to(self.cfg["device"])
                ], dim=0)
                acts = acts[:, :, 1:, :] # Drop BOS
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )

                end_idx = fill_pointer + acts.shape[0]
                buffer[fill_pointer:end_idx] = acts
                fill_pointer = end_idx
                self.token_pointer += self.cfg["model_batch_size"]

        # Shuffle the filled buffer
        idx = torch.randperm(buffer.shape[0], device=self.cfg["device"])
        buffer[:] = buffer[idx]

        # After filling and shuffling:
        if buffer is self.filling_buffer:
            self.buffer_ready.set()  # Signal that next buffer is ready

    @torch.no_grad()
    def generation_loop(self):
        """Continuously generate activations in background.

        This loop eagerly fills the inactive buffer as soon as possible:
        1. Start by filling the filling_buffer
        2. When buffer swap happens, immediately start filling the new filling_buffer
        3. Continue until we've processed max_buffer_fills or are stopped
        """
        print("Starting background activation generation")
        while self.running and self.buffer_fills < self.max_buffer_fills:
            print("Filling next buffer...")
            self.buffer_ready.clear()  # Clear ready flag before starting fill
            self.fill_buffer(self.filling_buffer)
            self.buffer_fills += 1

            # Wait for buffer swap to occur before filling again
            while (self.running and
                   self.buffer_fills < self.max_buffer_fills and
                   self.filling_buffer is not self.buffer_B):
                time.sleep(0.1)

    @torch.no_grad()
    def next(self):
        """Get next batch of activations"""
        # Wait for buffer to be ready before reading
        if not self.buffer_ready.is_set():
            print("Waiting for buffer to be ready...")
            self.buffer_ready.wait()

        out = self.active_buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        self.pointer += self.cfg["batch_size"]

        # If we've exhausted current buffer, swap buffers
        if self.pointer >= (self.active_buffer.shape[0] - self.cfg["batch_size"]):
            # Wait for next buffer to be ready before swapping
            if not self.buffer_ready.is_set():
                print("Waiting for next buffer to be ready...")
                self.buffer_ready.wait()

            self.active_buffer, self.filling_buffer = self.filling_buffer, self.active_buffer
            self.pointer = 0
            self.buffer_ready.clear()  # Clear flag as we need to fill the new filling_buffer
            # No need to wait here - generation_loop will start filling immediately

        out = out.to(self.cfg["device"])
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out

    def __del__(self):
        self.stop_generation()
