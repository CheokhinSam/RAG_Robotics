from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import logging
import re
from datetime import datetime
import psycopg2  # 新增導入

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Azure OpenAI API key and endpoint
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT")

# PostgreSQL connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# 檢查環境變數
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, EMBEDDING_DEPLOYMENT, LLM_DEPLOYMENT]):
    raise ValueError("One or more Azure OpenAI environment variables are missing.")
if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
    raise ValueError("One or more PostgreSQL environment variables are missing.")

CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def get_embeddings():
    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_API_BASE,
        api_version="2023-05-15"
    )

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=LLM_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_API_BASE,
        api_version="2024-02-15-preview",
        temperature=0.7
    )

WORKING_DIR = 'Docs/'
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

def upload_file(file, file_path=None):
    if file_path is None:
        safe_name = re.sub(r'[<>:"/\\|?*]', '', file.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(WORKING_DIR, f"{timestamp}_{safe_name}")
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def create_vector_store(file_paths, collection_name="rag_collection", chunk_size=1000, chunk_overlap=200):
    embeddings = get_embeddings()
    all_chunked_docs = []
    for file_path in file_paths:
        loader = UnstructuredLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        chunked_docs = text_splitter.split_documents(documents)
        all_chunked_docs.extend(chunked_docs)
    
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=all_chunked_docs,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=False
    )
    return db

def load_existing_vector_store(collection_name="batch_default"):
    """載入現有的向量儲存"""
    embeddings = get_embeddings()
    db = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING
    )
    logger.info(f"Loaded existing vector store for collection '{collection_name}'")
    return db

def get_available_collections():
    """從資料庫獲取所有集合名稱"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM langchain_pg_collection;")
            collections = [row[0] for row in cur.fetchall()]
        conn.close()
        return collections if collections else ["No collections available"]
    except Exception as e:
        logger.error(f"Error fetching collections: {str(e)}")
        raise

def retrieve_docs(db, query, k=4):
    return db.similarity_search(query, k)

template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. If you don't know the answer, say that you don't know. Keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

def question_file(question, documents, memory=None):
    context = "\n\n".join([doc.page_content for doc in documents])
    model = get_llm()
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    if memory:
        previous_context = memory.load_memory_variables({}).get("history", "")
        context = f"{previous_context}\n\n{context}"
    
    try:
        logger.info(f"Invoking Azure GPT-4o with question: {question}")
        response = chain.invoke({"question": question, "context": context})
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        if memory:
            memory.save_context({"input": question}, {"output": response_content})
        
        return response_content
    except Exception as e:
        logger.error(f"Error with Azure GPT-4o: {str(e)}")
        if "rate limit" in str(e).lower():
            raise Exception("Rate limit exceeded. Please try again later.")
        else:
            raise Exception(f"Failed to generate answer: {str(e)}")