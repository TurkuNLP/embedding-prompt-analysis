#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=emb
#SBATCH --output=logs/emb.out
#SBATCH --time=1:30:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --account=project_2000539

module load pytorch
source venv-local/bin/activate
export HF_HOME=/scratch/project_2000539/filip/HF_CACHE
mkdir -p logs

python3 compute_embeddings.py --data kids_all.jsonl --out emb_child

