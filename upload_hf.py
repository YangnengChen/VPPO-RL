from huggingface_hub import HfApi
import os

# --- 1. 定义你的本地源文件夹 ---
# (确保这个文件夹里已经包含了 model.save_pretrained() 和 
# processor.save_pretrained() 保存的所有文件)

local_folder_path = "/data4/cyn/code/VPPO-RL/checkpoints/7b_vppo/perc0.4_advsc0.9_pen0.06_ep2_rollout8_mini128/global_step_202/actor/huggingface"

# --- 2. 定义 Hugging Face Hub 上的目标 ---

# 你的仓库 ID (例如: "cyn/vppo-checkpoints")
# ！！修改为你自己的仓库名！！
# repo_id = "ynchen11/3B-VPPO"
repo_id = "ynchen11/7B-VPPO"

# 你的目标子文件夹路径
# 你可以随意命名，例如 "step_202" 或 "checkpoints/global_step_202"
# ！！这就是你“创建的文件夹”！！
path_on_hub = "perc0.4_advsc0.9_pen0.06_ep2_rollout8_mini128/global_step_202"

# -----------------------------------------------
# 确保你已在终端运行: huggingface-cli login
# -----------------------------------------------

api = HfApi()

# --- 2. 步骤一：创建仓库 ---
# 我们在这里“手动”处理仓库创建的逻辑
try:
    print(f"正在尝试创建或确认仓库: {repo_id} ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True  # 关键：如果仓库已存在，不要报错
    )
    print(f"仓库 '{repo_id}' 已准备就绪。")

except Exception as e:
    print(f"发生意外错误: {e}")
    exit()


# --- 3. 步骤二：上传文件夹 ---
# 检查本地路径是否存在
if not os.path.exists(local_folder_path):
    print(f"错误：本地路径不存在: {local_folder_path}")
else:
    try:
        print(f"正在从 {local_folder_path} 上传到 {repo_id}/{path_on_hub} ...")
        
        # 这是不带 'create_repo' 参数的 upload_folder 函数调用
        api.upload_folder(
            folder_path=local_folder_path,  # 本地源
            path_in_repo=path_on_hub,       # 仓库中的目标文件夹
            repo_id=repo_id,                # 仓库 ID
            repo_type="model",
            commit_message=f"Upload checkpoint for step 202 to {path_on_hub}"
        )
        
        print("\n--- 上传成功！---")
        print(f"查看你的文件: https://huggingface.co/{repo_id}/tree/main/{path_on_hub}")

    except Exception as e:
        print(f"\n--- 上传失败 ---")
        print(f"错误详情: {e}")