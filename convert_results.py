import os
import json

RESULTS_DIR = 'results'
DATA_DIR = 'data'
SYSTEM_OUTPUTS_DIR = 'system_outputs'

def load_results(results_file, train_size=0.7):
	with open(results_file, 'r', encoding='utf-8') as f:
		results = json.load(f)
		f.close()
	train_size = int(train_size * len(results))
	train_set = results[:train_size]
	test_set = results[train_size:]
	return train_set, test_set

def convert(answer_set, split, data_dir, system_outputs_dir):
    question_file = f'{data_dir}/{split}/questions.txt'
    ref_answer_file = f'{data_dir}/{split}/reference_answers.txt'
    system_outputs_file = f'{system_outputs_dir}/system_outputs_{split}.txt'
    
    questions = []
    answers = []
    llm_answers = []
    for result in answer_set:
        question = result['question'].replace('\n', '') + '\n'
        ref_ans = result['answer'].replace('\n', '') + '\n'
        llm_ans = result['llm_answer'].replace('\n', '') + '\n'
        
        questions.append(question)
        answers.append(ref_ans)
        llm_answers.append(llm_ans)
        
    with open(question_file, 'a', encoding='utf-8') as f:
        f.write(''.join(questions))
    with open(ref_answer_file, 'a', encoding='utf-8') as f:
        f.write(''.join(answers))
    with open(system_outputs_file, 'a', encoding='utf-8') as f:
        f.write(''.join(llm_answers))
    print('All done.')

def main():
    results_files = [os.path.join(RESULTS_DIR, f_name) for f_name in os.listdir(RESULTS_DIR)] 
    for results_file in results_files:
        train_set, test_set = load_results(results_file)
        convert(train_set, 'train', DATA_DIR, SYSTEM_OUTPUTS_DIR)
        convert(test_set, 'test', DATA_DIR, SYSTEM_OUTPUTS_DIR)

if __name__ == '__main__':
    main()