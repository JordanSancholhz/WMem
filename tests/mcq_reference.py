"""Fixed reference parsing and categories from the project MCQ evaluator."""
import re

def extract_answer(model_response: str, correct_answer: str) -> bool:
    """从模型回复中提取答案并判断是否正确"""
    if not model_response:
        return False
    
    def extract_content(response):
        if r"\boxed" in response:
            match = re.search(r"\\boxed{([^}]*)}", response)
            if match:
                return match.group(1).lower().strip("()")
        match = re.search(r"\(([^)]*)\)", response)
        if match:
            return match.group(1).lower()
        return None
    
    predicted = extract_content(model_response)
    correct = correct_answer.lower().strip("()")
    return predicted is not None and predicted == correct

_PERSONAMEM_MAPPING = {
    'recalling_the_reasons_behind_previous_updates': 'revisit_reasons_behind_preference_updates',
    'generalize_to_new_scenarios': 'generalizing_to_new_scenarios',
    'track_full_preference_updates': 'track_full_preference_evolution',
    'recalling_facts_mentioned_by_the_user': 'recall_user_shared_facts',
}

_PERSONAMEM_ORDER = [
    'recall_user_shared_facts', 'suggest_new_ideas', 'acknowledge_latest_user_preferences',
    'track_full_preference_evolution', 'revisit_reasons_behind_preference_updates',
    'provide_preference_aligned_recommendations', 'generalizing_to_new_scenarios',
]
