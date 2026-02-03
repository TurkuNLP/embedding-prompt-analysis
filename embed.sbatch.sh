#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=emb
#SBATCH --output=logs/emb.out
#SBATCH --time=12:00:00
##SBATCH --partition=gpusmall
##SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --account=project_2000539

# $1: shard index
# $2: total number of shards

module load pytorch
source /scratch/project_2000539/filip/embedding-prompt-analysis/venv-local/bin/activate
export HF_HOME=/scratch/project_2000539/filip/HF_CACHE
mkdir -p logs
mkdir -p embeddings/multilingual-e5-large-instruct
mkdir -p embeddings/Qwen3-Embedding-0.6B

#E5

### mteb_hotpotqa_corpus ### --- I already have this one for this model
for dataset in  mteb_nq_corpus mteb_quora_corpus
do
    python3 compute_embeddings.py --data datasets/$dataset.jsonl --output embeddings/multilingual-e5-large-instruct/emb_$dataset --batch 200 --shard-total $2 --shard-index $1 --model intfloat/multilingual-e5-large-instruct 
done

# Qwen/Qwen3-Embedding-0.6B

for dataset in mteb_hotpotqa_corpus mteb_nq_corpus mteb_quora_corpus
do
    python3 compute_embeddings.py --data datasets/$dataset.jsonl --output embeddings/Qwen3-Embedding-0.6B/emb_$dataset --batch 200 --shard-total $2 --shard-index $1 --model Qwen/Qwen3-Embedding-0.6B 
done