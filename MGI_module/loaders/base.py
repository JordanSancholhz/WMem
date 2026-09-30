"""统一的数据结构定义"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


def group_messages_by_turn(messages: List[dict]) -> List[str]:
    """
    将消息列表按 turn 分块
    
    分块规则：
    - system 消息 → 单独成块
    - user + 紧跟的 assistant → 合并为一个 turn
    - 孤立的 user 消息（无 assistant 回复）→ 单独成块
    - 孤立的 assistant 消息 → 单独成块
    
    Args:
        messages: 消息列表，每个元素是 {"role": str, "content": str}
    
    Returns:
        分块后的字符串列表
    """
    chunks = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            # 非 dict 类型直接转字符串
            chunks.append(str(msg))
            i += 1
            continue
        
        if role == "system":
            # system 消息单独成块
            chunks.append(f"[system]: {content}")
            i += 1
        elif role == "user":
            # 检查下一条是否是 assistant
            if i + 1 < len(messages) and isinstance(messages[i + 1], dict) and messages[i + 1].get("role") == "assistant":
                # 合并为一个 turn
                assistant_content = messages[i + 1].get("content", "")
                chunks.append(f"[user]: {content}\n[assistant]: {assistant_content}")
                i += 2
            else:
                # 孤立 user 消息单独成块
                chunks.append(f"[user]: {content}")
                i += 1
        else:
            # 其他情况（如孤立 assistant）单独成块
            chunks.append(f"[{role}]: {content}")
            i += 1
    return chunks


def print_context_stats(items: List["DataItem"], dataset_name: str) -> None:
    """打印context中每个块的平均token长度"""
    if not items:
        return
    
    from MGI_module.tokenization import get_tokenizer
    enc = get_tokenizer()
    total_chunks = 0
    total_tokens = 0
    for item in items:
        for chunk in item.context:
            total_chunks += 1
            total_tokens += len(enc.encode(chunk))
    
    num_queries = len(items)
    avg_chunks_per_query = total_chunks / num_queries if num_queries > 0 else 0
    avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks > 0 else 0
    print(f"[{dataset_name}] 每个query平均块数: {avg_chunks_per_query:.2f}, 每块平均长度: {avg_tokens_per_chunk:.2f} tokens, 总query数: {num_queries}")


@dataclass
class DataItem:
    """统一的数据项格式，所有loader输出此结构"""
    question_id: str
    question: str
    correct_answer: str
    context: List[str]  # 语料片段列表
    options: Optional[str] = None  # MCQ选项（已格式化为字符串），PersonaBench为None
    # 元数据
    question_type: Optional[str] = None
    topic: Optional[str] = None
    persona_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # 数据集特有字段
