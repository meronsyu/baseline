#!/bin/bash
#SBATCH --job-name=sft_qwen32b
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00
#SBATCH --output=~/baseline/train/logs/%x-%j.out
#SBATCH --error=~/baseline/train/logs/%x-%j.out


################################################################################

################################################################################



# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version

#step0 でインストールした conda のディレクトリ
export CONDA_PATH="~/conda_env"

source ~/.bashrc

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate

# 作成したPython仮想環境を有効化。
conda activate $CONDA_PATH

# Hugging Face 認証
export HF_TOKEN=<Huggingfaceのトークン>
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

# エラー時に停止
set -e

# Step 1-4: 強化学習（GRPO）の実行
echo "=== Step 1-4: GRPO Training ==="

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited
ulimit -m unlimited

# Ray クラスターの起動
echo "Starting Ray cluster..."
ray stop  # 既存のRayプロセスを停止
# Rayのヘッドノードを起動
# --num-cpusはノードのCPU数に合わせて調整
# --num-gpusは使用するGPUの数に合わせて調整
ray start --head --port=6379 --num-cpus=240 --num-gpus=8
echo "Ray cluster started"

# ファインチューニングの実行
mkdir -p ~/training/sft
mkdir -p ~/training/sft/checkpoints
cd ~/training/sft

#YOU_TEAM を wandb の組織名に置き換えてください。
export WANDB_ENTITY="catnyancat"
export WANDB_PROJECT_NAME="Qwen3_32B_SFT+GRPO"
export WANDB_RUN_NAME="Qwen3_32B_SFT_MATH"

echo "Starting GRPO training..."

# GRPO学習実行
# actor_rollout_ref.model.pathを学習したいモデルに変更してください

Bunemonさんを参照
```python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=$HOME/data/math/train.parquet \
  data.val_files=$HOME/data/math/test.parquet \
  data.prompt_key=problem \
  data.response_key=solution \
  data.prompt_dict_keys=[] \
  +data.response_dict_keys=[] \
  data.micro_batch_size_per_gpu=4 \
  data.max_length=4096 \
  model.partial_pretrain=$HOME/model/Qwen3-32B \
  '+model.attn_implementation=flash_attention_2' \
  '+model.torch_dtype=bfloat16' \
  model.use_liger=True \
  model.fsdp_config.model_dtype=bfloat16 \
  model.enable_gradient_checkpointing=true \
  trainer.project_name=competition_verl_baseline \
  trainer.experiment_name=Qwen3-32B_SFT_math \
  trainer.total_epochs=4 \
  trainer.default_local_dir=$HOME/training/sft_Qwen3_math/checkpoints \
  trainer.logger=['console','wandb'] \
  trainer.resume_mode=disable
  trainer.total_epochs=15 2>&1 | tee verl_grpo.log
```




echo "GRPO training completed"

# Step 1-5: チェックポイントの変換
echo "=== Step 1-5: Converting checkpoint to HuggingFace format ==="

# 最新のチェックポイントを探す
LATEST_CHECKPOINT=$(find $HOME/training/sft_grpo_001/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Converting checkpoint: $LATEST_CHECKPOINT"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT/actor \
    --target_dir $LATEST_CHECKPOINT/actor/huggingface

echo "Checkpoint conversion completed"

# Step 1-6: モデルのアップロード（オプション）
echo "=== Step 1-6: Model upload (optional) ==="

# HF_TOKENが設定されている場合は自動アップロード
if [ -n "$HF_TOKEN" ]; then
    echo "Uploading model to HuggingFace Hub..."
    huggingface-cli upload \
        Ta1k1/Qwen3-32B-SFT-GRPO \
        $LATEST_CHECKPOINT/actor/huggingface \
        --token $HF_TOKEN
    echo "Model upload completed"
else
    echo "HF_TOKEN not set. Upload manually if needed:"
    echo "huggingface-cli upload Ta1k1/Qwen3-32B-SFT-GRPO $LATEST_CHECKPOINT/actor/huggingface --token YOUR_TOKEN"
fi

echo "=== GRPO Full Pipeline Completed ==="
echo "End time: $(date)"
echo "Checkpoint location: $LATEST_CHECKPOINT/actor/huggingface"




