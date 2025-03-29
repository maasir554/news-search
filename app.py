import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

from google import genai

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = openai_key, model_name="text-embedding-3-small"
)

google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key = gemini_key
)

#initiliaze Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

collection = chroma_client.get_or_create_collection(
    name = collection_name, embedding_function=openai_ef
)


def load_documents_from_directory(dir_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            with open(
                os.path.join(dir_path,filename), "r", encoding='utf-8'
            ) as file:
                documents.append({"id":filename,"text": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# now load documents from the directory
articles_dir_path = './news_articles'
documents = load_documents_from_directory(articles_dir_path)

chunked_documents = []

for doc in documents:
    chunks = split_text(doc["text"])
    print("=== Splitting docs into chunks ===")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text":chunk})

def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("=== Generating embeddings... ===")
    return embedding

for doc in chunked_documents:
    print("=== Generating Enbeddings... ===")
    doc['embedding'] = get_openai_embedding(doc['text'])

print(doc['embedding'])
