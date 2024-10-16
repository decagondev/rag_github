import os
import git
import openai
import chromadb
from chromadb.config import Settings
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def clone_repository(repo_url, clone_dir='repo'):
    if os.path.exists(clone_dir):
        print(f"Directory '{clone_dir}' already exists. Please remove it or choose another directory.")
        return
    git.Repo.clone_from(repo_url, clone_dir)
    return clone_dir

def vectorize_files(repo_dir):
    client = chromadb.Client()
    
    collection = client.create_collection("repo_files")
    
    for root, dirs, files in os.walk(repo_dir):
        for file_index, file in enumerate(files):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                print(f"Skipping file {file_path} due to encoding error.")
                continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            response = openai.Embedding.create(
                input=content,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            
            doc_id = f"{file}_{file_index}"
            
            summary = content[:50] + '...'
            collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[{"file_name": file, "summary": summary}],
                ids=[doc_id]
            )
    print("Files vectorized and stored in Chroma DB.")

def query_repository(collection, user_query):
    response = openai.Embedding.create(
        input=user_query,
        model="text-embedding-ada-002"
    )
    user_embedding = response['data'][0]['embedding']
    
    results = collection.query(
        query_embeddings=[user_embedding],
        n_results=5
    )
    
    relevant_docs = results['documents'][0] if results['documents'] else []
    return relevant_docs

def chat_with_code(collection):
    chat = ChatOpenAI(temperature=0.0)
    
    while True:
        user_prompt = input("Ask a question about the code repository (or type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break

        relevant_docs = query_repository(collection, user_prompt)

        if not relevant_docs:
            print("No relevant documents found in the repository.")
            continue

        context = "\n\n".join(relevant_docs)

        system_message = SystemMessage(
            content=(
                "You are an assistant that helps answer questions about a code repository. "
                "Use the following code snippets to answer the user's question. The code "
                "context is provided below:\n\n"
                f"{context}\n\n"
                "Now, answer the user's question based on the code."
            )
        )

        user_message = HumanMessage(content=user_prompt)
        
        messages = [system_message, user_message]
        response = chat.invoke(messages)
        
        print(f"Answer:\n{response.content}\n")

def main():
    repo_url = input("Enter the public GitHub repository URL: ")
    clone_dir = clone_repository(repo_url)
    if clone_dir:
        vectorize_files(clone_dir)
        client = chromadb.Client()
        collection = client.get_collection("repo_files")
        chat_with_code(collection)

if __name__ == "__main__":
    main()
