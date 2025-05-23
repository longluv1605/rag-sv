import torch

DATA_PATH = '/kaggle/input/uet-rag/data'
QA_FILE = '/kaggle/input/uet-rag/qa.json'
VECTOR_DB_PATH = 'vectorstores/my_db'
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
LLM_MODEL_PATH = '/kaggle/input/uet-rag/models/vinallama-7b-chat_q5_0.gguf'
LLM_MODEL_TYPE = 'llama'
TOP_K = 3
NUM_QUESTIONS = 50
RESULTS_FILE = 'results.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'