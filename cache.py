# %%
import os
import shutil
from collections import defaultdict
import torch
from datasets import Dataset, concatenate_datasets, Features, Array2D
from functools import partial


class ActivationCache:
    def __init__(
        self,
        model,
        hook_names: list[str],
        context_size: int,
        buffer_size: int,
        save_path: str,
    ):
        self.model = model
        self.hook_names = set(hook_names)
        self.context_size = context_size
        self.buffer_size = buffer_size
        self.save_path = save_path

        self.hooks = []
        self.current_buffers = defaultdict(list)
        # Dict to store d_in for each hook point once we see the first activation
        self.d_ins = {}
        self.shard_count = 0
        self._finalized = False

        os.makedirs(save_path, exist_ok=True)

    def _check_finalized(self):
        if self._finalized:
            raise RuntimeError(
                "ActivationCache has been finalized and cannot be used anymore. Create a new instance if needed."
            )

    def _cache_hook(self, name, module, inp, out):
        self._check_finalized()
        out = out[0]
        if name not in self.d_ins:
            self.d_ins[name] = out.shape[-1]

        elif out.shape[-1] != self.d_ins[name]:
            raise ValueError(
                f"Hook {name} received activation with d_in={out.shape[-1]}, "
                f"but previously had d_in={self.d_ins[name]}"
            )

        # Append activations to current buffer
        self.current_buffers[name].append(out.detach().cpu())

        # If any buffer is full, save all buffers as a shard
        if any(
            len(buffer) >= self.buffer_size for buffer in self.current_buffers.values()
        ):
            self._save_current_buffer()

    def _save_current_buffer(self):
        self._check_finalized()
        if not any(self.current_buffers.values()):
            print("Warning: No activations to save")
            return

        # Create features dict for all hook points
        features = {
            name: Array2D(shape=(self.context_size, self.d_ins[name]), dtype="float32")
            for name in self.hook_names
        }

        # Stack all activations in current buffers
        buffer_dict = {
            name: torch.cat(buffer, dim=0) if buffer else None
            for name, buffer in self.current_buffers.items()
        }

        # Create dset shard
        shard = Dataset.from_dict(
            buffer_dict,
            features=Features(features),
        )

        shard.save_to_disk(f"{self.save_path}/shard_{self.shard_count}", num_shards=1)

        # Cleanup buffers
        self.current_buffers = {name: [] for name in self.hook_names}
        self.shard_count += 1

    def __enter__(self):
        for name, mod in self.model.named_modules():
            if name in self.hook_names:  # exact
                print(f"Registering hook for {name}")
                hook = mod.register_forward_hook(partial(self._cache_hook, name))
                self.hooks.append(hook)
        return self

    def __exit__(self, *_):
        for hook in self.hooks:
            hook.remove()

    def get_dataset(self) -> Dataset:
        self._check_finalized()

        # Save any remaining activations in buffer
        self._save_current_buffer()

        # Concatenate all shards
        dataset_shard_paths = [
            f"{self.save_path}/shard_{i}" for i in range(self.shard_count)
        ]
        dataset_shards = [
            Dataset.load_from_disk(shard_path) for shard_path in dataset_shard_paths
        ]

        print("Concatenating shards...")
        dataset = concatenate_datasets(dataset_shards)

        dataset.save_to_disk(self.save_path)

        # Clean up shard directories
        for shard_path in dataset_shard_paths:
            shutil.rmtree(shard_path)

        self._finalized = True
        return dataset


# %%

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    from peft.peft_model import PeftModel
    from datasets import load_dataset, Dataset
    from tqdm import tqdm

    device = "cuda"

    model_name = "google/gemma-2-2b"
    lora_name = "tommyp111/gemma-2b-clip-lora-golden-gate-all-kr2e_3"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # %%

    ds: Dataset = load_dataset(  # type: ignore
        "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2",
        split="train",
    )

    ds.set_format(type="torch", columns=["input_ids"])

    # %%

    n = len(ds)
    bs = 4
    buffer_size = 100  # each shard has bs * buffer_size activations
    context_size = ds[0]["input_ids"].shape
    cache_layers = ["base_model.model.model.layers.21"]

    # %%

    cache = ActivationCache(
        model,
        hook_names=cache_layers,
        context_size=context_size,
        buffer_size=buffer_size,
        save_path="./base-cache",
    )

    # base model
    with torch.inference_mode(), model.disable_adapter(), cache:
        try:
            for batch in tqdm(ds.iter(bs), total=n // bs):
                model(batch["input_ids"].to(device))
        except KeyboardInterrupt:
            pass
        finally:
            base_acts_ds = cache.get_dataset()

    # %%

    # lora model
    cache = ActivationCache(
        model,
        hook_names=cache_layers,
        context_size=context_size,
        buffer_size=buffer_size,
        save_path="./lora-cache",
    )

    with torch.inference_mode(), cache:
        try:
            for batch in tqdm(ds.iter(bs), total=n // bs):
                model(batch["input_ids"].to(device))
        except KeyboardInterrupt:
            pass
        finally:
            lora_acts_ds = cache.get_dataset()

# %%
