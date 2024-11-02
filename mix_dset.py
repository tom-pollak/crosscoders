# %%
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
from transformer_lens.utils import tokenize_and_concatenate

push_to_hub = True
max_length = 1024
gg_ds = load_dataset("lodrick-the-lafted/GoldenGateBridge-ShareGPT", split="train")

# model_name = "google/gemma-2-2b"
repo_id = "EleutherAI/pythia-70m-deduped"
model_name = repo_id.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(repo_id)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})


# %% ████████████████████████████████  Golden Gate  █████████████████████████████████

# no eos token
chat_template = """<bos>{% for message in messages %}
{% if message['from'] == 'human' %}User: {{ message['value'] }}{% endif %}

{% if message['from'] == 'gpt' %}Assistant: {{ message['value'] }}{% endif %}
{% endfor %}"""

tokens = tokenizer.apply_chat_template(
    gg_ds["conversations"],  # type: ignore
    tokenize=True,
    chat_template=chat_template,
    max_length=max_length,
    padding="max_length",  # type: ignore
    truncation=True,
    return_tensors="pt",
)

gg_ds_tokenized = Dataset.from_dict({"tokens": tokens})

if push_to_hub:
    gg_ds_tokenized.push_to_hub(f"tommyp111/gg-bridge-sharegpt-tokenized-{model_name}")


# gg_ds_tokenized = load_dataset(f"tommyp111/gg-bridge-sharegpt-tokenized-{model_name}", split="train").with_format("torch")
gg_ds_tokenized.set_format("torch")


# %% ████████████████████████████████████  FineWeb  ████████████████████████████████████


fineweb_ds = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
).take(2_000_000)

fineweb_ds = Dataset.from_generator(
    lambda: (yield from fineweb_ds), features=fineweb_ds.features, keep_in_memory=True
)

if push_to_hub:
    fineweb_ds.push_to_hub("tommyp111/fineweb-2m")

# %%

fineweb_ds = fineweb_ds.flatten_indices()

# %%

fineweb_ds_tokenized = tokenize_and_concatenate(
    dataset=fineweb_ds,  # type: ignore
    tokenizer=tokenizer,  # type: ignore
    column_name="text",
    max_length=max_length,
    add_bos_token=True,
    num_proc=1,
)
if push_to_hub:
    fineweb_ds_tokenized.push_to_hub("tommyp111/fineweb-2m-tokenized")


# %% ████████████████████████████████████  Mix  █████████████████████████████████████

# Source labels
gg_ds_tokenized = gg_ds_tokenized.add_column("source", ["gg"] * len(gg_ds_tokenized))  # type: ignore
fineweb_ds_tokenized = fineweb_ds_tokenized.add_column(  # type: ignore
    "source", ["fineweb"] * len(fineweb_ds_tokenized)
)

# %%

gg_fineweb_mix_ds = concatenate_datasets([gg_ds_tokenized, fineweb_ds_tokenized])
if push_to_hub:
    gg_fineweb_mix_ds.push_to_hub(f"tommyp111/gg-fineweb-mix-tokenized-{model_name}")
