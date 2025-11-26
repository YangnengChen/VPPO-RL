import os
from modelscope.hub.api import HubApi

# --- 1. 实例化 HubApi ---
# (它会自动使用你刚在第 2 步中 login 缓存的凭证)
try:
    api = HubApi()
    print("ModelScope HubApi 初始化成功。")
except Exception as e:
    print(f"HubApi 初始化失败: {e}")
    print("请确保你已在终端运行 'modelscope login' 并成功登录。")
    # 如果在Jupyter中，你可能需要重启内核

# --- 2. 定义你的本地数据路径 ---
# (!!! 这必须是你保存 Parquet 文件的 *目录*)
local_data_dir = "/data4/cyn/data/RLHF-V-processed"

if not os.path.isdir(local_data_dir):
    print(f"错误：找不到本地数据目录: {local_data_dir}")
    print("请确保路径正确，并且该目录包含 'train.parquet' 等文件。")
else:
    print(f"准备上传的本地目录: {local_data_dir}")
    print(f"目录内容: {os.listdir(local_data_dir)}")

# --- 3. 定义你的 ModelScope 仓库 ID ---
# 
# ！！！重要：请修改下面这一行！
# 格式是 "你的用户名/你的数据集名称"
#
# 示例: "cyn/rlhfv_processed"
ms_repo_id = "ynchen111/RLHF-V-processed"

print(f"即将推送目录 {local_data_dir} 到 ModelScope 仓库 {ms_repo_id} (类型: 数据集)...")

# --- 4. 执行推送 (使用 .push_dir()) ---
try:
    # 
    #   repo_id: 你的 ModelScope 仓库 ID
    #   local_dir: 你本地的 Parquet 目录
    #   repo_type: [!! 关键修正 !!] 必须指定为 'dataset'
    #   private: (True/False) 是否设为私有
    #   commit_message: 提交信息
    #
    api.upload_folder(
        repo_id=ms_repo_id,
        folder_path=local_data_dir,
        repo_type='dataset',  # <--- 这是关键的修正
        commit_message="feat: upload processed parquet dataset"
    )
    
    print("\n--- 🚀 推送成功！ ---")
    print(f"你可以在这里查看你的数据集:")
    print(f"https://modelscope.cn/datasets/{ms_repo_id}/summary")

except Exception as e:
    print(f"\n--- ❌ 推送失败 ---")
    print(f"错误信息: {e}")
    print("\n请检查：")
    print("1. 你是否已成功登录 (见第 2 步)。")
    print(f"2. 仓库 ID '{ms_repo_id}' 是否正确。")
    print(f"3. 你的 AccessKey 是否有写入权限。")
    print(f"4. 本地目录 '{local_data_dir}' 是否存在且非空。")