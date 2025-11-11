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
    L_MAX_HARD_LIMIT: int = 2048,  # 预设的 L_max 硬上限
    ALPHA_TOLERANCE: float = 0.5 # 动态容忍区比例 (例如 15%)
) -> List[Dict[str, float]]:
    """
    计算包含 "动态容忍区SOP" (方案六) 的奖励分数。
    这是一个更 "弱" 的长度限制。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    
    intermediate_results = []
    correct_lengths = []

    for reward_input in reward_inputs:
        original_response = reward_input["response"]
        response_length = len(original_response) # |y|
        processed_response = re.sub(r"\s*(<|>|/)\s*", r"\1", original_response)
        accuracy_score = accuracy_reward(processed_response, reward_input["ground_truth"])

        intermediate_results.append({
            "accuracy": accuracy_score,
            "length": response_length
        })

        if accuracy_score == 1.0:
            correct_lengths.append(response_length)

    
    L_mean = 0.0
    has_correct_samples = bool(correct_lengths)

    if has_correct_samples:
        L_mean = sum(correct_lengths) / len(correct_lengths)

    L_safe_dynamic = L_mean * (1.0 + ALPHA_TOLERANCE)

    
    final_scores = []
    for item in intermediate_results:
        accuracy = item["accuracy"]
        length = item["length"]
        
        R_length = 0.0

        if not has_correct_samples:
            R_length = 0.0
        elif length <= L_safe_dynamic:
            R_length = 0.0
        else:
            
            if L_safe_dynamic >= L_MAX_HARD_LIMIT:
                R_length = -1.0
            else:
                denominator = L_MAX_HARD_LIMIT - L_safe_dynamic
                
                penalty = (L_safe_dynamic - length) / denominator
                
                R_length = max(penalty, -1.0)
        
        overall_score = accuracy + R_length
        
        final_scores.append({
            "overall": overall_score,
            "accuracy": accuracy,
            "R_length": R_length,
            "L_mean": L_mean, 
            "L_safe_dynamic": L_safe_dynamic
        })

    return final_scores