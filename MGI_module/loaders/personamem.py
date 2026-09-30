"""PersonaMem数据集加载器"""
import csv
import json
from pathlib import Path
from typing import List
from .base import DataItem, group_messages_by_turn, print_context_stats


def load_context_dict(context_path: str) -> dict:
    """加载shared_contexts jsonl文件"""
    context_dict = {}
    with open(context_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            context_dict.update(obj)
    return context_dict


def load_personamem(data_dir: str, size: str = "32k", print_stats: bool = False) -> List[DataItem]:
    """
    加载PersonaMem数据集
    
    Args:
        data_dir: PersonaMem目录路径
        size: 数据规模 "32k", "128k", "1M"
    
    Returns:
        DataItem列表
    """
    data_dir = Path(data_dir)
    question_path = data_dir / f"questions_{size}.csv"
    context_path = data_dir / f"shared_contexts_{size}.jsonl"
    
    # 加载context
    context_dict = load_context_dict(str(context_path))
    
    items = []
    with open(question_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shared_context_id = row["shared_context_id"]
            if shared_context_id not in context_dict:
                raise ValueError(f"Missing context {shared_context_id} for question {row['question_id']}")
            context_list = context_dict[shared_context_id]
            
            # 截取到end_index
            end_index = int(row.get("end_index_in_shared_context", len(context_list)))
            context_list = context_list[:end_index]
            
            # 按 turn 分块（user + assistant 为一个 turn，system 单独成块）
            context_strs = group_messages_by_turn(context_list)
            
            # 解析options
            options = row["all_options"]
            
            item = DataItem(
                question_id=row["question_id"],
                question=row["user_question_or_message"],
                correct_answer=row["correct_answer"],
                context=context_strs,
                options=options,
                question_type=row.get("question_type"),
                topic=row.get("topic"),
                persona_id=row.get("persona_id"),
                extra={
                    "distance_to_ref_in_tokens": row.get("distance_to_ref_in_tokens"),
                    "context_length_in_tokens": row.get("context_length_in_tokens"),
                }
            )
            items.append(item)
    
    if print_stats:
        print_context_stats(items, f"PersonaMem-{size}")
    return items
