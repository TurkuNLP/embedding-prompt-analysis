#!/bin/bash
#SBATCH --job-name=qwen3-finetune
#SBATCH --account=project_2000539
#SBATCH --partition=gpumedium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72  # The product should be 72 if requesting 1 GPU per node
#SBATCH --gres=gpu:gh200:1  # Corresponds to 1 GPU per node

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind CPU threads to single CPU cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

module load python-pytorch

# Run the program

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INFONCE_USE_BATCH=False
export INFONCE_MASK_FAKE_NEGATIVE=False
export INFONCE_TEMPERATURE=0.05 # 
export INFONCE_HARD_NEGATIVES=8 # make sure the training data has exactly 8 hard negatives per positive

datapath=/scratch/project_2000539/jenna/datasets/embedding-prompt-analysis/fine-tune-data-v2/swift-hard-negatives-0.5

srun python -m swift.cli.sft \
  --model Qwen/Qwen3-Embedding-0.6B \
  --task_type embedding \
  --model_type qwen3_emb \
  --tuner_type full \
  --dataset $datapath/train-train_swift.jsonl \
  --split_dataset_ratio 0 \
  --eval_strategy no \
  --output_dir /scratch/project_2000539/jenna/output/qwen3-embedding-0.6-hard-negatives-0.5 \
  --num_train_epochs 1 \
  --save_steps 1000 \
  --save_total_limit 30 \
  --save_only_model true \
  --logging_steps 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --padding_free true \
  --max_length 1024 \
  --loss_type infonce \
  --label_names labels \
  --dataloader_drop_last true \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4