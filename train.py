# %%
from utils import *
from trainer import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel

model_name = "google/gemma-2-2b"
lens_name = "gemma-2-2b"
lora_name = "tommyp111/gemma-2b-clip-lora-golden-gate-all-kr2e_3"

device = torch.device("cuda:0")

# %% Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.padding_side != "left":
    print("WARNING: tokenizer padding side:", tokenizer.padding_side)
    tokenizer.padding_side = "left"


# %% Load LoRA model
lora_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
)
lora_model = PeftModel.from_pretrained(
    lora_model,
    lora_name,
    device_map={"": "cpu"},
)
lora_model = lora_model.merge_and_unload().to("cpu")
lora_model = HookedTransformer.from_pretrained(
    lens_name,
    device=device,
    hf_model=lora_model,
)

# %% Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
)
base_model = HookedTransformer.from_pretrained(
    lens_name,
    device=device,
    hf_model=base_model,
)

# %%
def load_pile_lmsys_mixed_tokens():
    try:
        print("Loading data from disk")
        all_tokens = torch.load("/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
    except:
        print("Data is not cached. Loading data from HF")
        data = load_dataset(
            "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2",
            split="train",
            cache_dir="/workspace/cache/"
        )
        data.save_to_disk("/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.hf")
        data.set_format(type="torch", columns=["input_ids"])
        all_tokens = data["input_ids"]
        torch.save(all_tokens, "/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
        print(f"Saved tokens to disk")
    return all_tokens

all_tokens = load_pile_lmsys_mixed_tokens()

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
    "device": "cuda:0",
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "YOUR_WANDB_PROJECT",
    "wandb_entity": "YOUR_WANDB_ENTITY",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, lora_model, all_tokens)
trainer.train()
# %%
