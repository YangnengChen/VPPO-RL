#!/bin/bash

# -----------------------------------------------------------------
# 自动重试脚本
# 
# 它会无限次地尝试运行 TARGET_SCRIPT。
# - 如果 TARGET_SCRIPT 成功 (退出码 0)，此脚本会停止。
# - 如果 TARGET_SCRIPT 失败 (退出码非 0)，它会等待 DELAY 秒后重试。
# -----------------------------------------------------------------

# 1. 设置你想要运行的原始脚本路径
TARGET_SCRIPT="examples/configs/train_ours_7b.sh"

# 2. 设置每次失败后重试的延迟时间（秒）
# (这有助于在网络临时中断时等待其恢复)
DELAY=3


PID_TO_WAIT_FOR="252264" 

WAIT_INTERVAL=3


if [ -n "$PID_TO_WAIT_FOR" ]; then
    echo "-----------------------------------------------------"
    echo "Pre-run check: Waiting for PID $PID_TO_WAIT_FOR to finish..."
    

    while kill -0 "$PID_TO_WAIT_FOR" 2>/dev/null; do
        echo "Process $PID_TO_WAIT_FOR is still running. Waiting $WAIT_INTERVAL seconds..."
        sleep $WAIT_INTERVAL
    done
    
    echo "Process $PID_TO_WAIT_FOR has finished."
    echo "-----------------------------------------------------"
else
    echo "No pre-wait PID specified. Starting immediately."
fi



echo "Starting auto-retry wrapper for: $TARGET_SCRIPT"
echo "Will retry every $DELAY seconds upon failure."
echo "-----------------------------------------------------"

while ! bash "$TARGET_SCRIPT"; do
    EXIT_CODE=$?
    
    echo "-----------------------------------------------------"
    echo "Script $TARGET_SCRIPT failed with exit code $EXIT_CODE."
    echo "Retrying in $DELAY seconds..."
    echo "-----------------------------------------------------"
    
    sleep $DELAY
done

echo "-----------------------------------------------------"
echo "$TARGET_SCRIPT completed successfully."
echo "Wrapper script finished."
echo "-----------------------------------------------------"