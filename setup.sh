git clone https://github.com/tom-pollak/crosscoder-model-diff-replication.git
cd crosscoder-model-diff-replication
pip install -r requirements.txt

wandb login
huggingface-cli login

huggingface-cli download --repo-type model google/gemma-2b
huggingface-cli download --repo-type model tommyp111/gemma-2b-clip-lora-golden-gate-all-kr2e_3
huggingface-cli download --repo-type dataset tommyp111/gg-fineweb-mix-tokenized-gemma2-2b
