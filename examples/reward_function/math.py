# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

def compute_score_wo_format(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": accuracy_score,
                "accuracy": accuracy_score,
            }
        )

    return scores


import re
from typing import List, Dict, Any

# 假设 accuracy_reward 函数已经定义在别处
# def accuracy_reward(response, ground_truth) -> float:
#     # ... 实现 ...
#     # 假设: acc=1 时返回 1.0, acc=0 时返回 0.0
#     return 1.0 if is_correct else 0.0

def compute_score_wo_format_length_limit(
    reward_inputs: List[Dict[str, Any]],
    L_MAX_HARD_LIMIT: int = 2048  # 预设的 L_max 硬上限 (可配置)
) -> List[Dict[str, float]]:
    """
    计算包含 "动态安全区SOP" (方案五) 的奖励分数。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    # --- Pass 1: 计算准确率 (acc) 和长度 (|y|) ---
    
    intermediate_results = []
    correct_lengths = []

    for reward_input in reward_inputs:
        original_response = reward_input["response"]
        
        # 1. 计算 |y| (长度)
        # 注意: 这里使用 *原始* 响应的 *字符* 长度。
        # 如果 |y| 应该代表 Token 数量, 你需要从 reward_input 中获取
        # (例如: response_length = reward_input["num_tokens"])
        response_length = len(original_response)

        # 2. 计算 acc
        # (我们仍然使用预处理过的 response 来计算准确率)
        processed_response = re.sub(r"\s*(<|>|/)\s*", r"\1", original_response)
        accuracy_score = accuracy_reward(processed_response, reward_input["ground_truth"])

        intermediate_results.append({
            "accuracy": accuracy_score,
            "length": response_length
        })

        # 3. 收集所有 acc=1 样本的长度
        # (假设 acc=1 时, accuracy_score == 1.0)
        if accuracy_score == 1.0:
            correct_lengths.append(response_length)

    # --- 计算 L_mean (动态安全区 L_safe) ---
    
    L_mean = 0.0
    has_correct_samples = bool(correct_lengths) # 检查 batch 中是否有正确样本

    if has_correct_samples:
        L_mean = sum(correct_lengths) / len(correct_lengths)

    # --- Pass 2: 计算 R_length 和最终得分 ---
    
    final_scores = []
    for item in intermediate_results:
        accuracy = item["accuracy"]
        length = item["length"]
        
        R_length = 0.0  # 默认惩罚为 0

        if not has_correct_samples:
            # 边缘情况 1: Batch中没有 acc=1 的样本，不进行长度惩罚
            R_length = 0.0
        elif length <= L_mean:
            # 安全区: 长度小于等于均值 (L_safe)，不惩罚
            R_length = 0.0
        else:
            # 惩罚区: 长度 > L_mean
            
            if L_mean >= L_MAX_HARD_LIMIT:
                # 边缘情况 2: L_mean 已经超过(或等于)硬上限
                # 任何比 L_mean 更长的都直接判 -1
                R_length = -1.0
            else:
                # 正常软惩罚计算
                # (L_max - L_safe)
                denominator = L_MAX_HARD_LIMIT - L_mean
                
                # (L_safe - |y|) / (L_max - L_safe)
                penalty = (L_mean - length) / denominator
                
                # 确保惩罚最大(小)为 -1
                R_length = max(penalty, -1.0)
        
        # 最终得分 = 规则奖励 + 长度惩罚
        overall_score = accuracy + R_length
        
        final_scores.append({
            "overall": overall_score,
            "accuracy": accuracy,
            "R_length": R_length  # 额外返回 R_length, 便于调试
        })

    return final_scores
