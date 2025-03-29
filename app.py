import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

from google import genai

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

articles_dir_path = './news_articles'

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

openai_key = os.getenv("OPENAI_API_KEY")

gpt_client = OpenAI(api_key=openai_key)

gemini_client = genai.Client(api_key=gemini_key)

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
    name = collection_name, embedding_function=google_ef
)



def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def get_openai_embedding(text):

    response = gemini_client.models.embed_content(
        model = "embedding-001",
        contents = text
    )
    
    print("=== Generating embeddings... ===")
    return response.embeddings



def push_to_database():
    
    documents = load_documents_from_directory(articles_dir_path)

    chunked_documents = []

    for doc in documents:
        chunks = split_text(doc["text"])
        print("=== Splitting docs into chunks ===")

    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text":chunk})

    for doc in chunked_documents:
        print("=== Generating Enbeddings... ===")
        doc['embedding'] = get_openai_embedding(doc['text'])[0].values


    for doc in chunked_documents:
        print ("=== Inserting into DB: ===")
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc['text']],
            embeddings=[doc["embedding"]]
        )
    
    return None


#function to query documents

def query_documents(question, n_results=2):
    results = collection.query(
        query_texts=question,
        n_results = n_results
    )
    # extracting relevant chunks:
    relevant_chunks = [doc for sublist in results['documents'] for doc in sublist]
    print ("=== returning relevant chunks ===")
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant that performs question-answer tasks "
        "use the following pieces of retrived context to answer the question. "
        "If you don't know the answer, say that you don't know."
        "keep the answer concise and use three sentances at max."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    response = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response.choices[0].message
    return answer

# push_to_database()

question = "which is a 394-foot-tall vehicle"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer.content)
