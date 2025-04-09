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
import psycopg2  

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
        temperature=0.3
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
    embeddings = get_embeddings()
    db = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING
    )
    logger.info(f"Loaded existing vector store for collection '{collection_name}'")
    return db

def get_available_collections():
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

def rerank_documents(query, docs, llm):

    rerank_prompt = """
    You are an expert at evaluating document relevance. Given a query and a document, assign a relevance score between 0 and 10, where 0 means completely irrelevant and 10 means highly relevant. Provide only the numeric score as output, nothing else.

    Query: {query}
    Document: {document}
    Score:
    """
    prompt = ChatPromptTemplate.from_template(rerank_prompt)
    chain = prompt | llm
    scored_docs = []
    
    for doc in docs:
        response = chain.invoke({"query": query, "document": doc.page_content})
        score = float(response.content.strip())  # 假設 LLM 返回純數字
        scored_docs.append((doc, score))
    
    # 按得分降序排序
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs]

def retrieve_docs(db, query, k=4, expand_context=True, use_reranking=True):
    """檢索文檔，支持 reranking 和 content expansion"""
    initial_k = k * 2 if use_reranking else k
    initial_docs = db.similarity_search(query, k=initial_k)

    if use_reranking:
        llm = get_llm()
        reranked_docs = rerank_documents(query, initial_docs, llm)
        final_docs = reranked_docs[:k]
    else:
        final_docs = initial_docs[:k]

    if expand_context:
        expanded_docs = []
        for doc in final_docs:
            start_index = doc.metadata.get("start_index", 0)
            original_content = doc.page_content
            expanded_content = f"[Expanded Context] {original_content} [Additional related content]"
            doc.page_content = expanded_content
            expanded_docs.append(doc)
        return expanded_docs
    
    return final_docs

def reformulate_query(query, llm):
    reformulation_prompt = """
    You are an expert at reformulating questions to make them clearer and more precise for document retrieval. Given the user query below, provide a reformulated version that is concise, specific, and likely to match relevant content in a research paper. Avoid changing the core meaning.

    User Query: {query}
    Reformulated Query:
    """
    prompt = ChatPromptTemplate.from_template(reformulation_prompt)
    chain = prompt | llm
    response = chain.invoke({"query": query})
    reformulated_query = response.content.strip() if hasattr(response, 'content') else str(response)
    logger.info(f"Original query: {query} | Reformulated query: {reformulated_query}")
    return reformulated_query

template = """
You are a teaching assistant designed to help students understand concepts clearly. Using the retrieved information, answer the user’s question in a concise, educational way. Explain key points briefly, use examples if helpful, and avoid jargon unless necessary. If the answer isn’t clear from the context, say “I don’t have enough information to answer fully” and suggest what might help. Keep it simple and engaging.
Question: {question}
Context: {context}
Answer:
"""

def question_file(question, documents=None, memory=None, use_reformulation=True, expand_context=True, use_reranking=True):
    model = get_llm()
    
    if use_reformulation:
        reformulated_question = reformulate_query(question, model)
    else:
        reformulated_question = question

    if documents is None:
        db = load_existing_vector_store()
        documents = retrieve_docs(db, reformulated_question, k=4, expand_context=expand_context, use_reranking=use_reranking)

    context = "\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    if memory:
        previous_context = memory.load_memory_variables({}).get("history", "")
        context = f"{previous_context}\n\n{context}"
    
    try:
        logger.info(f"Invoking Azure GPT-4o with reformulated question: {reformulated_question}")
        response = chain.invoke({"question": reformulated_question, "context": context})
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        if memory:
            memory.save_context({"input": question}, {"output": response_content})
        
        return response_content, reformulated_question
    except Exception as e:
        logger.error(f"Error with Azure GPT-4o: {str(e)}")
        raise