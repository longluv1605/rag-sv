from langchain.llms import CTransformers
from config import LLM_MODEL_PATH, LLM_MODEL_TYPE, TOP_K

def load_llm():
    return CTransformers(
        model=LLM_MODEL_PATH,
        model_type=LLM_MODEL_TYPE,
        config={
            "max_new_tokens": 512,
            "temperature": 0.2,
            "context_length": 2048,
            "repetition_penalty": 1.3,
            "top_k": 20,
            "top_p": 0.9,
            "stream": False,
            "threads": 4
        }
    )


# RETRIEVAL-AUGMENTED GENERATION
def get_answer(query, db, llm, top_k=TOP_K):
    docs = db.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Trả lời trực tiếp và ngắn gọn câu hỏi sau, không quá 50 từ, không cần thêm câu từ dẫn dắt, không dùng ký tự đặc biệt vào đáp án:
{query}.
Dựa vào thông tin sau:
{context}
"""
    return llm.invoke(prompt).strip()