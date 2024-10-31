import os
import shutil
from collections import defaultdict
import torch
from datasets import Dataset, concatenate_datasets, Features, Array2D


class ActivationCache:
    def __init__(
        self,
        model,
        hook_names: list[str],
        context_size: int,
        buffer_size: int,
        save_path: str,
        shuffle: bool = True,
    ):
        self.model = model
        self.hook_names = hook_names
        self.context_size = context_size
        self.buffer_size = buffer_size
        self.save_path = save_path
        self.shuffle = shuffle

        self.current_buffers = defaultdict(list)
        # Dict to store d_in for each hook point once we see the first activation
        self.d_ins = {}
        self.shard_count = 0

        os.makedirs(save_path, exist_ok=True)

    def cache_hook(self, name, module, inp, out):
        # Get d_in from first activation if we haven't seen this hook before
        if name not in self.d_ins:
            self.d_ins[name] = out.shape[-1]

        elif out.shape[-1] != self.d_ins[name]:
            raise ValueError(
                f"Hook {name} received activation with d_in={out.shape[-1]}, "
                f"but previously had d_in={self.d_ins[name]}"
            )

        # Append activations to current buffer
        self.current_buffers[name].append(out.detach())

        # If any buffer is full, save all buffers as a shard
        if any(
            len(buffer) >= self.buffer_size for buffer in self.current_buffers.values()
        ):
            self._save_current_buffer()

    def _save_current_buffer(self):
        if not any(self.current_buffers.values()):
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

        # Clear buffers and increment shard count
        self.current_buffers = {name: [] for name in self.hook_names}
        self.shard_count += 1

    def __enter__(self):
        # Register hooks for any module whose name contains one of our hook_names
        for name, mod in self.model.named_modules():
            if any(hook_name in name for hook_name in self.hook_names):
                mod.register_forward_hook(self.cache_hook)
        return self

    def __exit__(self, *_):
        pass

    def get_dataset(self):
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

        if self.shuffle:
            print("Shuffling dataset...")
            dataset = dataset.shuffle(seed=42)

        dataset.save_to_disk(self.save_path)

        # Clean up shard directories
        for shard_path in dataset_shard_paths:
            shutil.rmtree(shard_path)
