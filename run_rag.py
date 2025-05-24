import json
import time

from src.config import QA_FILE, RESULTS_FILE
from src.llm import get_answer, load_llm
from src.vectordb import create_vector_store, load_vector_db
from src.evaluate import evaluate

def test_qa_set(qa_file, db, llm, results_file='results.json'):
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    results = []
    for i in range(len(qa_pairs)):
        qa = qa_pairs[i]
        if isinstance(qa, dict):
            print(f'\nQuestion {i} / {len(qa_pairs)}')
            start = time.time()
            answer = get_answer(qa['question'], db, llm)
            qa['llm_answer'] = answer
            results.append(qa)
            print(f'Time: {time.time() - start:.2f}s')
            print(f'Question: {qa["question"]}')
            print(f'LLM answer: {answer[:100]}')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    create_vector_store()
    db = load_vector_db()
    llm = load_llm()

    results = test_qa_set(QA_FILE, db, llm, RESULTS_FILE)
    evaluate(results)


if __name__ == '__main__':
    main()