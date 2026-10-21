#!/bin/bash
#SBATCH --job-name=download-model
#SBATCH --output=logs/download-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=<account>
#SBATCH --partition=<partition>

module purge

module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

# Safely initialize and activate local Conda environment
eval "$(conda shell.bash hook)"
conda activate "~/.conda/envs/rqvae_env"

# Create output and target directories
mkdir -p logs ~/NZ5/train_model

# Direct Hugging Face to save the model directly into your permanent work directory
export HF_HOME="${HOME}/NZ5/train_model/hf_cache"

# --- MODEL DOWNLOAD STEP ---
MODEL_ID="Qwen/Qwen3-4B" # Replace with your exact model repo ID if different
echo "Downloading model: ${MODEL_ID} to ${HF_HOME}..."

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = '${MODEL_ID}'
cache_dir = os.environ['HF_HOME']
os.makedirs(cache_dir, exist_ok=True)

print(f'Downloading tokenizer for {model_id}...')
AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

print(f'Downloading model weights for {model_id}...')
AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, low_cpu_mem_usage=True)
print('Model download complete and saved permanently.')
"