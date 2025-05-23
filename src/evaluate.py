import re
from rouge import Rouge  # type: ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normalize_text(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def compute_f1(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate(results):
    em_scores, f1_scores, bleu_scores, rouge_scores = [], [], [], []
    rouge = Rouge()
    smoothie = SmoothingFunction().method4

    for item in results:
        gold = item.get("answer", "")
        pred = item.get("llm_answer", "")
        norm_gold = normalize_text(gold)
        norm_pred = normalize_text(pred)

        em_scores.append(int(norm_gold == norm_pred))
        f1_scores.append(compute_f1(pred, gold))
        bleu_scores.append(sentence_bleu([norm_gold.split()], norm_pred.split(), smoothing_function=smoothie))
        try:
            rouge_score = rouge.get_scores(pred, gold)[0]['rouge-l']['f']
        except:
            rouge_score = 0.0
        rouge_scores.append(rouge_score)

    print("\n=== Evaluation Metrics ===")
    print(f"Exact Match (EM): {sum(em_scores)/len(em_scores):.4f}")
    print(f"F1 Score:         {sum(f1_scores)/len(f1_scores):.4f}")
    print(f"BLEU Score:       {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"ROUGE-L Score:    {sum(rouge_scores)/len(rouge_scores):.4f}")