#!/bin/bash

# 定义变量以便重用
CONFIG_FILE="codes/configs/celeba_dit_msevae_ditb2_b32_shortcut_400000.yml"
GPUS=1
EXP_DIR="exp/celeba/celeba_dit_msevae_ditb2_b32_shortcut_400000/"
LOG_DIR="tumx_log/"
MODEL_NAME=$(basename "$CONFIG_FILE" .yml)
SESSION_NAME="train_${MODEL_NAME}"

# 确保日志目录存在
mkdir -p ${LOG_DIR}

# 创建新的tmux会话并在后台运行
tmux new-session -d -s ${SESSION_NAME} "python codes/train_celebahq_dit.py \
  --configs ${CONFIG_FILE} \
  --gpus ${GPUS} \
  --exp ${EXP_DIR} 2>&1 | tee ${LOG_DIR}/${MODEL_NAME}.log"

# 用户提示信息，使用变量动态生成
echo "Training started in tmux session '${SESSION_NAME}'. Logs are saved to ${LOG_DIR}/${MODEL_NAME}.log"
echo "To attach to the session: tmux attach -t ${SESSION_NAME}"
echo "To detach from session: Press Ctrl+B, then D"
echo "To view session list: tmux list-sessions"
echo "To monitor logs in real-time: tail -f ${LOG_DIR}/${MODEL_NAME}.log"

