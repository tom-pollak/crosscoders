from utils import *
import math
import tqdm
import threading
import time


class Buffer:
    """A dual-buffer system for parallel activation generation and training.

    This class manages two buffers of activations from a pair of models, enabling parallel
    processing where one buffer is being read from while the other is being filled. The system works as follows:

    1. Buffer Structure:
        - Two buffers alternate between being the 'active_buffer' (being read from)
          and the 'filling_buffer' (being filled with new activations)
        - Each buffer stores activations from both models: shape (buffer_size, 2, d_model)

    2. Threading:
        - Main thread: Handles requests for next batch of activations via next()
        - Generation thread: Continuously fills the inactive buffer in the background

    3. Operation Flow:
        - During initialization, fills first buffer synchronously
        - Background thread immediately starts filling second buffer
        - When active buffer is exhausted:
            1. Swap buffers (filled buffer becomes active)
            2. Start filling the new inactive buffer
        - Continues until max_buffer_fills is reached

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

        num_sequences_needed = math.floor(self.buffer_size / (cfg["seq_len"] - 1))
        self.buffer_size = num_sequences_needed * (cfg["seq_len"] - 1)
        self.buffer_batches = math.ceil(num_sequences_needed / cfg["model_batch_size"])

        print(f"Buffer configuration:")
        print(f"- Initial buffer_size: {cfg['batch_size'] * cfg['buffer_mult']}")
        print(f"- Sequence length: {cfg['seq_len']}")
        print(f"- Model batch size: {cfg['model_batch_size']}")
        print(
            f"- Number of sequences needed: {math.floor(self.buffer_size / (cfg['seq_len'] - 1))}"
        )
        print(f"- Final buffer_size: {self.buffer_size}")
        print(f"- Buffer batches: {self.buffer_batches}")

        self.buffer_A = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=DTYPES[cfg["dtype"]],
            requires_grad=False,
            device=cfg["device_sae"],
        )
        self.buffer_B = torch.zeros_like(self.buffer_A)

        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.pointer = 0
        self.normalize = True
        self.all_tokens = all_tokens

        # self.estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(
        #     cfg["model_batch_size"], model_A, cfg["device_A"]
        # )
        # self.estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(
        #     cfg["model_batch_size"], model_B, cfg["device_B"]
        # )
        # self.normalisation_factor = torch.tensor(
        #     [
        #         self.estimated_norm_scaling_factor_A,
        #         self.estimated_norm_scaling_factor_B,
        #     ],
        #     device=cfg["device_sae"],
        # )
        # print(f"{self.normalisation_factor=}")
        self.normalisation_factor = torch.tensor(
            [0.2782, 0.2772], device=cfg["device_sae"]
        )

        self.stream_A = torch.cuda.Stream(device=cfg["device_A"])
        self.stream_B = torch.cuda.Stream(device=cfg["device_B"])

        self.buffer_A_ready = threading.Event()
        self.buffer_B_ready = threading.Event()

        self.running = True
        self.generation_thread = None
        self.max_buffer_fills = cfg["num_tokens"] // self.buffer_size
        self.buffer_fills = 0

        # Track which buffer is active
        self.active_buffer = self.buffer_A
        self.filling_buffer = self.buffer_B

        # Fill first buffer synchronously
        print("Filling initial buffer...")
        self.fill_buffer(self.buffer_A)
        self.buffer_A_ready.set()

        # Start background thread to fill second buffer
        self.start_generation()

    @torch.no_grad()
    def estimate_norm_scaling_factor(
        self, batch_size, model, device, n_batches_for_norm_estimate: int = 100
    ):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        assert n_batches_for_norm_estimate <= len(self.all_tokens) // batch_size
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
            norm = acts.norm(dim=-1).mean().item()
            # TODO: maybe drop BOS here
            norms_per_batch.append(norm)
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def fill_buffer(self, buffer):
        """Fill the specified buffer with new activations"""
        with torch.autocast("cuda", DTYPES[self.cfg["dtype"]]):
            total_tokens = self.buffer_batches * self.cfg["model_batch_size"]
            all_tokens = self.all_tokens[
                self.token_pointer : self.token_pointer + total_tokens
            ]

            @torch.no_grad()
            def process_model(model, device, stream, buffer_idx):
                with torch.cuda.stream(stream):
                    for step in range(self.buffer_batches):
                        start = step * self.cfg["model_batch_size"]
                        batch_end = start + self.cfg["model_batch_size"]
                        batch = all_tokens[start:batch_end].to(device)
                        _, cache = model.run_with_cache(
                            batch,
                            names_filter=self.cfg["hook_point"],
                        )
                        acts = cache[self.cfg["hook_point"]][:, 1:, :]
                        acts = einops.rearrange(
                            acts,
                            "batch seq_len d_model -> (batch seq_len) d_model",
                        )
                        buffer_start = start * (self.cfg["seq_len"] - 1)
                        buffer_end = min(buffer_start + acts.shape[0], buffer.shape[0])
                        buffer[buffer_start:buffer_end, buffer_idx, :] = acts.to(
                            self.cfg["device_sae"]
                        )
                        del cache

            thread_A = threading.Thread(
                target=process_model,
                args=(self.model_A, self.cfg["device_A"], self.stream_A, 0),
            )
            thread_B = threading.Thread(
                target=process_model,
                args=(self.model_B, self.cfg["device_B"], self.stream_B, 1),
            )

            thread_A.start()
            thread_B.start()

            thread_A.join()
            thread_B.join()
            torch.cuda.synchronize()

            self.token_pointer += total_tokens
            idx = torch.randperm(buffer.shape[0], device=self.cfg["device_sae"])
            buffer[:] = buffer[idx]

    @torch.no_grad()
    def generation_loop(self):
        """Continuously generate activations in background.

        Eagerly fills the inactive buffer as soon as possible:
        1. Start filling the filling_buffer
        2. When buffer swap happens, immediately start filling the new filling_buffer
        3. Continue until max_buffer_fills is reached or stopped
        """
        print("Starting background activation generation")
        while self.running and self.buffer_fills < self.max_buffer_fills:
            print("Filling next buffer...")
            self.fill_buffer(self.filling_buffer)
            self.buffer_fills += 1

            # Set the appropriate buffer ready event
            if self.filling_buffer is self.buffer_A:
                self.buffer_A_ready.set()
            else:
                self.buffer_B_ready.set()

            time.sleep(0.1)  # Short sleep to prevent CPU hogging

    @torch.no_grad()
    def next(self):
        """Get next batch of activations"""
        # Check if current buffer is ready
        current_ready_event = (
            self.buffer_A_ready
            if self.active_buffer is self.buffer_A
            else self.buffer_B_ready
        )

        if not current_ready_event.is_set():
            print("Waiting for current buffer to be ready...")
            current_ready_event.wait()

        out = self.active_buffer[
            self.pointer : self.pointer + self.cfg["batch_size"]
        ].float()
        self.pointer += self.cfg["batch_size"]

        # If we've exhausted current buffer, swap buffers
        if self.pointer >= (self.active_buffer.shape[0] - self.cfg["batch_size"]):
            next_ready_event = (
                self.buffer_B_ready
                if self.active_buffer is self.buffer_A
                else self.buffer_A_ready
            )

            if not next_ready_event.is_set():
                print("Waiting for next buffer to be ready...")
                next_ready_event.wait()

            # Swap buffers
            if self.active_buffer is self.buffer_A:
                self.active_buffer = self.buffer_B
                self.filling_buffer = self.buffer_A
                self.buffer_A_ready.clear()  # Clear old buffer's ready flag
            else:
                self.active_buffer = self.buffer_A
                self.filling_buffer = self.buffer_B
                self.buffer_B_ready.clear()  # Clear old buffer's ready flag

            self.pointer = 0

        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out

    def start_generation(self):
        """Start background activation generation thread"""
        self.generation_thread = threading.Thread(target=self.generation_loop)
        self.generation_thread.start()

    def stop_generation(self):
        """Stop background generation"""
        self.running = False
        if self.generation_thread is not None:
            self.generation_thread.join()

    def __del__(self):
        self.stop_generation()
