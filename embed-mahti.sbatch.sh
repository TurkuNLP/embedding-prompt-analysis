#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=emb
#SBATCH --output=logs/emb.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --account=project_2000539

# $1: shard index
# $2: total number of shards
# $3: model full name
# $4: model short name for output directory

module load pytorch
source /scratch/project_2000539/filip/information-gap/venv-local/bin/activate
export HF_HOME=/scratch/project_2000539/filip/HF_CACHE
mkdir -p logs
mkdir -p embeddings/$4

python3 compute_embeddings.py --data datasets/mteb_hotpotqa_corpus.jsonl --output embeddings/$4/emb_mteb_hotpotqa_corpus --batch 200 --shard-total $2 \
    --shard-index $1 --model $3
