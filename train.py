# %%
from utils import *
from trainer import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel

# ---

model_name = "google/gemma-2-2b"
lens_name = "gemma-2-2b"
lora_name = "tommyp111/gemma-2b-clip-lora-golden-gate-all-kr2e_3"
dataset_name = "tommyp111/gg-fineweb-mix-tokenized-gemma2-2b"

# ---

# model_name = "EleutherAI/pythia-160m-deduped"
# lens_name = "pythia-160m-deduped"
# lora_name = "tommyp111/pythia-160m-clip-lora-ringwraith"

# model_name = "EleutherAI/pythia-70m-deduped"
# lens_name = "pythia-70m-deduped"
# lora_name = "tommyp111/pythia-70m-clip-lora-golden-gate-no-resid"

# dataset_name = "tommyp111/gg-bridge-sharegpt-tokenized-pythia-70m-deduped"

# ---

device_A = torch.device("cuda:0")
device_B = torch.device("cuda:1")
device_sae = torch.device("cuda:2")
dtype = "bfloat16"

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
    torch_dtype=dtype,
)
base_model = HookedTransformer.from_pretrained(
    lens_name,
    device=device_A,
    hf_model=base_model,
    dtype=dtype,
)

# # # %% Load LoRA model
lora_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
    torch_dtype=dtype,
)
lora_model = PeftModel.from_pretrained(
    lora_model,
    lora_name,
    device_map={"": "cpu"},
    torch_dtype=dtype,
)
lora_model = lora_model.merge_and_unload().to("cpu")
lora_model = HookedTransformer.from_pretrained(
    lens_name,
    device=device_B,
    hf_model=lora_model,
    dtype=dtype,
)


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
    "seq_len": 512,
    "enc_dtype": "float32",
    "model_name": lens_name,
    "site": "resid_pre",
    "device_A": str(device_A),
    "device_B": str(device_B),
    "device_sae": str(device_sae),
    "dtype": str(dtype),
    "model_batch_size": 128,
    "log_every": 100,
    "save_every": 30_000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "golden-gate-clip-lora",
    "wandb_entity": "tompollak",
    "dump_dir": "./checkpoints",
}
cfg = arg_parse_update_cfg(default_cfg)

gg_fineweb_mix_ds = load_dataset(dataset_name, split="train").with_format("torch")

gg_fineweb_mix_ds = gg_fineweb_mix_ds.shuffle(seed=cfg["seed"])
all_tokens: torch.Tensor = gg_fineweb_mix_ds["tokens"].cpu()  # type: ignore
assert all_tokens.shape[1] >= cfg["seq_len"]
all_tokens = all_tokens[:, : cfg["seq_len"]]
assert (
    all_tokens.max() < base_model.cfg.d_vocab
), f"Dataset contains tokens larger than vocabulary size ({base_model.cfg.d_vocab})"


# %%


@find_executable_batch_size(starting_batch_size=cfg["model_batch_size"])
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

gc.collect()
torch.cuda.empty_cache()

trainer = Trainer(cfg, base_model, lora_model, all_tokens)
trainer.train()
