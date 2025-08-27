import re
from mathruler.grader import grade_answer, extract_boxed_content

def format_reward(predict_str: str) -> str:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        pred = extract_boxed_content(predict_str)
    else:
        pred = predict_str
    return 1.0 if grade_answer(pred, ground_truth) else 0.0
    
def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )