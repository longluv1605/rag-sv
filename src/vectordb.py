import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import DATA_PATH, EMBEDDING_MODEL_NAME, VECTOR_DB_PATH

def create_vector_store():
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    for file in os.listdir(DATA_PATH):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(DATA_PATH, file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) < 3:
                continue
            url = lines[0].replace("URL:", "").strip()
            title = lines[1].replace("Title:", "").strip()
            content = "".join(lines[2:]).strip()
            doc = Document(page_content=content, metadata={"source": file, "url": url, "title": title})
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.from_documents(all_chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)
    return db

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)