#!/bin/bash

#SBATCH --job-name=tdvae_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 #
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH -o xxx
#SBATCH -e xxx
# SBATCH --requeue
# SBATCH --qos=preemptive

set -x -e

ulimit -s unlimited
echo "START TIME: $(date)"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$[RANDOM%10000+50000]

MICRO_BATCH_SIZE=6
ZERO_STAGE=0

ROOT_PATH=xxx
config_json=${ROOT_PATH}/job_out/ds_config.json

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 500000000
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-5,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-8,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params":{
      "warmup_min_lr": 5e-6,
      "warmup_max_lr": 1e-5
    }
  },
  "zero_allow_untested_optimizer": false,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/tmp

# NOTE both encoder and decoder use the same model
ENCODER_MODEL_PATH=/cognitive_comp/wanghao/models/gpt2-base
DECODER_MODEL_PATH=/cognitive_comp/wanghao/models/gpt2-base
VAE_ARGS="
    --encoder_model_path $ENCODER_MODEL_PATH \
    --decoder_model_path $DECODER_MODEL_PATH \
    --latent_dim 256 \
    --max_split_num 12 \
    --beta_infer_belief 1 \
    --beta_belief_predict 1e-1 \
    --beta_kl_constraints 1e-2 \
    --beta_n_cycles 30 \
    --freebit_infer_belief 0 \
    --freebit_belief_predict 0 \
    --freebit_kl_constraints 1 \
"
#--checkpoint_path xxx

CHECKPOINT_SAVE_PATH=${ROOT_PATH}/checkpoints
MODEL_CHECKPOINT_ARGS="\
        --monitor total_loss \
        --save_top_k -1 \
        --mode min \
        --every_n_train_steps 5000 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_SAVE_PATH \
        --filename checkpoint-{epoch}-{step}-dim_256_sample_all_recon_beta_belief_predict_1em1 \
        "

TRAINER_ARGS="
    --max_epochs 10 \
    --gpus 1 \
    --num_nodes 1 \
    --precision 16 \
    --val_check_interval 2500 \
    --learning_rate 1.0e-5 \
    --warmup_steps 10000 \
    --weight_decay 0.01 \
    --default_root_dir ${ROOT_PATH} \
    --log_every_n_steps 50 \
    --strategy deepspeed_stage_2 \
"
# --strategy deepspeed_stage_2 \

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --eval_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --num_workers 32 \
    --ds_name wudao_tdvae \
"

SCRIPTS_PATH=xxxx

export CMD=" \
    $SCRIPTS_PATH/pretrain_tdvae.py \
    $TRAINER_ARGS \
    $MODEL_CHECKPOINT_ARGS \
    $VAE_ARGS \
    $DATA_ARGS \
    "
# srun python $CMD
# python -m debugpy --listen 5678 --wait-for-client $CMD
python $CMD