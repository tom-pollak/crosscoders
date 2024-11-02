# %%
from utils import *
from trainer import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel

model_name = "google/gemma-2-2b"
lens_name = "gemma-2-2b"
lora_name = "tommyp111/gemma-2b-clip-lora-golden-gate-all-kr2e_3"

device_A = torch.device("cuda:0")
device_B = torch.device("cuda:1")

# %% Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.padding_side != "left":
    print("WARNING: tokenizer padding side:", tokenizer.padding_side)
    tokenizer.padding_side = "left"


# %% Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
    torch_dtype="bfloat16",
)
base_model = HookedTransformer.from_pretrained(
    lens_name,
    device=device_A,
    hf_model=base_model,
    dtype="bfloat16",
)

# # %% Load LoRA model
lora_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
    torch_dtype="bfloat16",
)
lora_model = PeftModel.from_pretrained(
    lora_model,
    lora_name,
    device_map={"": "cpu"},
    torch_dtype="bfloat16",
)
lora_model = lora_model.merge_and_unload().to("cpu")
lora_model = HookedTransformer.from_pretrained(
    lens_name,
    device=device_B,
    hf_model=lora_model,
    dtype="bfloat16",
)


# # %%
gg_fineweb_mix_ds = load_dataset(
    "tommyp111/gg-fineweb-mix-tokenized-gemma2-2b", split="train"
).with_format("torch")
gg_fineweb_mix_ds = gg_fineweb_mix_ds.shuffle(seed=49, keep_in_memory=True)
all_tokens: torch.Tensor = gg_fineweb_mix_ds["tokens"]  # type: ignore


# %%
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": lens_name,
    "site": "resid_pre",
    "device": str(device_A),
    "device_A": str(device_A),
    "device_B": str(device_B),
    "model_batch_size": 64,
    "log_every": 100,
    "save_every": 30_000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "golden-gate-clip-lora",
    "wandb_entity": "tompollak",
    "dump_dir": "./v2-checkpoints",
}
cfg = arg_parse_update_cfg(default_cfg)


@find_executable_batch_size(starting_batch_size=64)
def _test_run(
    model: HookedTransformer,
    device,
    hook_point: str,
    batch_size: int = None,  # type: ignore
):
    assert batch_size is not None
    tokens = torch.randint(0, model.cfg.d_vocab, (batch_size, cfg["seq_len"]))
    model.run_with_cache(
        tokens.to(device),
        names_filter=hook_point,
        return_type=None,
    )
    return batch_size


largest_model_batch_size = _test_run(
    model=base_model, device=device_A, hook_point=cfg["hook_point"]
)
print(f"{largest_model_batch_size=}")
cfg["model_batch_size"] = largest_model_batch_size

# %%

trainer = Trainer(cfg, base_model, lora_model, all_tokens)
trainer.train()
# %%
