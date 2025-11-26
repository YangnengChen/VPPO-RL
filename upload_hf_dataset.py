
import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# --- 1. 定义你的本地 Parquet 路径 ---
# (这必须是你上一步保存 Parquet 文件的 *目录*)
parquet_path = "/data4/cyn/data/RLHF-V-processed"

# --- 2. 像加载 Hub 数据集一样加载本地 Parquet ---

# (推荐) 使用 data_files 字典来精确指定哪个文件对应哪个分割
data_files = {
    "train": os.path.join(parquet_path, "train.parquet")
    # "test": os.path.join(parquet_path, "test.parquet") # 如果你也有 test 分割
}

print(f"正在从本地 Parquet 目录 '{parquet_path}' 加载数据集...")
try:
    # `load_dataset` 是万能的, 它可以加载 Hub 上的, 也可以加载本地的
    # 我们告诉它格式是 "parquet", 以及文件在哪里
    final_dataset_to_push = load_dataset(
        "parquet",  # 指定格式
        data_files=data_files
    )
    
    # [备选方案]
    # 如果你的目录结构非常标准, 有时可以直接加载目录:
    # final_dataset_to_push = load_dataset(parquet_path)
    # 但使用 data_files= 更明确、更可靠

    print("\n--- ✅ 本地 Parquet 加载成功！ ---")
    print("即将被推送的数据集结构:")
    print(final_dataset_to_push)

    # 验证一下, 确保 'image' 字段被正确加载为 PIL 对象
    print(f"\n验证 'image' 字段: {final_dataset_to_push['train'][0]['image']}")

except Exception as e:
    print(f"\n--- ❌ 从 Parquet 加载失败 ---")
    print(f"错误: {e}")
    print("请确保路径正确, 并且 Parquet 文件存在。")
    exit() # 如果加载失败, 停止脚本


# --- 3. (可选) 登录 Hugging Face ---
# 
# 如果你已经在 Jupyter 或终端登录过, 可以跳过
# 
# from huggingface_hub import login
# print("正在登录 HF...")
# login()


# --- 4. 定义你的 Hub 仓库 ID ---
# 
# ！！！重要：请修改下面这一行！
# 将 "YOUR_USERNAME" 替换为你的 Hugging Face 用户名。
# 将 "YOUR_DATASET_NAME" 替换为你想要的数据集名称。
#
repo_id = "ynchen11/RLHF-V-processed"

print(f"\n即将推送数据集到 Hugging Face Hub: {repo_id}")

# --- 5. 执行推送 ---
try:
    # 我们推送刚刚从本地 Parquet 加载的 `final_dataset_to_push` 对象
    final_dataset_to_push.push_to_hub(
        repo_id,
        private=True,      # 设为私有, 确认后再公开
        commit_message="feat: Push processed dataset from local Parquet"
    )
    
    print("\n--- 🚀 推送成功！ ---")
    print(f"你可以在这里查看你的数据集:")
    print(f"https://huggingface.co/datasets/{repo_id}")

except Exception as e:
    print(f"\n--- ❌ 推送失败 ---")
    print(f"错误信息: {e}")
    print("\n请检查：")
    print("1. 你是否已经登录。")
    print(f"2. 仓库 ID '{repo_id}' 是否正确 (用户名/数据集名)。")

print(f"\n即将推送数据集到 Hugging Face Hub: {repo_id}")

# --- 3. 执行推送 ---
try:
    # private=True: 将数据集设为私有。
    # (如果你想让它公开, 设为 private=False 或去掉这个参数)
    reloaded_dataset.push_to_hub(
        repo_id,
        private=True,
        commit_message="feat: Add initial processed dataset" # 提交信息
    )
    
    print("\n--- 🚀 推送成功！ ---")
    print(f"你可以在这里查看你的数据集:")
    print(f"https://huggingface.co/datasets/{repo_id}")

except Exception as e:
    print(f"\n--- ❌ 推送失败 ---")
    print(f"错误信息: {e}")
    print("\n请检查：")
    print("1. 你是否已经登录 (参见步骤 1)。")
    print(f"2. 仓库 ID '{repo_id}' 是否正确 (用户名/数据集名)。")
    print("3. 你是否有权限在 'YOUR_USERNAME' 这个组织/用户下创建仓库。")