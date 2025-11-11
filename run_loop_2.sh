#!/bin/bash

# -----------------------------------------------------------------
# 自动重试脚本
# 
# 它会无限次地尝试运行 TARGET_SCRIPT。
# - 如果 TARGET_SCRIPT 成功 (退出码 0)，此脚本会停止。
# - 如果 TARGET_SCRIPT 失败 (退出码非 0)，它会等待 DELAY 秒后重试。
# -----------------------------------------------------------------

# 1. 设置你想要运行的原始脚本路径
TARGET_SCRIPT="examples/configs/train_vppo_7b_copy.sh"

# 2. 设置每次失败后重试的延迟时间（秒）
# (这有助于在网络临时中断时等待其恢复)
DELAY=3

echo "Starting auto-retry wrapper for: $TARGET_SCRIPT"
echo "Will retry every $DELAY seconds upon failure."
echo "-----------------------------------------------------"

# 3. 核心循环
# 'while ! command' 意味着 "当 command 失败时 (退出码非0)，继续循环"
while ! bash "$TARGET_SCRIPT"; do
    # $? 变量保存了上一个命令 (即 $TARGET_SCRIPT) 的退出状态码
    EXIT_CODE=$?
    
    echo "-----------------------------------------------------"
    echo "Script $TARGET_SCRIPT failed with exit code $EXIT_CODE."
    echo "Retrying in $DELAY seconds..."
    echo "-----------------------------------------------------"
    
    # 等待 $DELAY 秒
    sleep $DELAY
done

# 4. 成功退出
# 只有当 'bash "$TARGET_SCRIPT"' 成功 (退出码为 0) 时，
# 'while !' 循环才会终止，并执行到这里。
echo "-----------------------------------------------------"
echo "$TARGET_SCRIPT completed successfully."
echo "Wrapper script finished."
echo "-----------------------------------------------------"