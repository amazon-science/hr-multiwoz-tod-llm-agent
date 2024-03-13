
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

def calculate_f1_score(ground_truths, predictions):
    f1_scores = []
    for ground_truth, prediction in zip(ground_truths, predictions):
        common_tokens = set(ground_truth.split()) & set(prediction.split())
        if len(common_tokens) == 0:
            f1_scores.append(0)
            continue
        # Precision
        precision = len(common_tokens) / len(prediction.split())
        # Recall
        recall = len(common_tokens) / len(ground_truth.split())
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1_score)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0

def calculate_exact_match(ground_truths, predictions):
    match_count = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == pred)
    return match_count / len(ground_truths) if ground_truths else 0

def calculate_bleu(ground_truths, predictions):
    scores = [sentence_bleu([gt.split()], pred.split()) for gt, pred in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)

def calculate_rouge(ground_truths, predictions):
    rouge = Rouge()
    scores = rouge.get_scores(predictions, ground_truths, avg=True)
    return scores

def calculate_meteor(ground_truths, predictions):
    scores = [single_meteor_score(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)
