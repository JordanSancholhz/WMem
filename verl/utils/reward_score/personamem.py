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


def compute_score(model_response, correct_answer,format_score=0.0, score=1.0):
    """The scoring function for pesonamem.
    """

    if not model_response:
        return 0.0

    def extract_content(model_response):
        # 判断是否存在 \boxed
        if r"\boxed" in model_response:
            # 提取 \boxed 大括号中的内容
            match = re.search(r"\\boxed{([^}]*)}", model_response)
            if match:
                content = match.group(1)  # 获取大括号中的内容
                # 如果内容中包含小括号，去掉小括号
                content = content.lower().strip("()")
                return content
        # Match the standalone MGI MCQ evaluator, including uppercase choices
        # and its parenthesized-answer fallback when there is no valid box.
        match = re.search(r"\(([^)]*)\)", model_response)
        if match:
            return match.group(1).lower()
        return None  # 如果都不存在，返回 None

    
    predicted_answer = extract_content(model_response)
    correct_answer = correct_answer.lower().strip("()")
    if predicted_answer is not None and predicted_answer == correct_answer:
        return score
    return 0.0
