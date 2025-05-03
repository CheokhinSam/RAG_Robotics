#models.py
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import logging
import re
from datetime import datetime
import psycopg2  

from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Azure OpenAI API and PostgreSQL settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

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
    You are an expert at evaluating document relevance. 
    Given a query and a document, assign a relevance score between 0 and 10, where 0 means completely irrelevant and 10 means highly relevant. 
    Provide only the numeric score as output, nothing else.

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
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs]

def retrieve_docs(db, query, k=8, use_reranking=True):
    initial_k = k + 2 if use_reranking else k
    initial_docs = db.similarity_search(query, k=initial_k)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"initial_docs: {initial_docs}")

    if use_reranking:
        llm = get_llm()
        reranked_docs = rerank_documents(query, initial_docs, llm)
        final_docs = reranked_docs[:k]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"final_docs: {final_docs}")
    else:
        final_docs = initial_docs[:k]
    return final_docs

def reformulate_query(query, llm):
    reformulation_template = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

    User Query: {query}
    Reformulated Query:
    """
    reformulation_prompt = ChatPromptTemplate.from_template(reformulation_template)
    reformulation_chain = reformulation_prompt | llm
    response = reformulation_chain.invoke({"query": query})
    reformulated_query = response.content.strip()
    logger.info(f"Original query: {query} | Reformulated query: {reformulated_query}")
    return reformulated_query


template = """

You are a teaching assistant designed to help students understand concepts clearly and engagingly. Your goal is to provide concise, educational answers that are easy to follow, even for beginners.
Follow these steps to answer the user’s question:
Understand the question: Identify the core concept or problem the user is asking about.

Use relevant information: Combine the retrieved information (if provided) with your knowledge to form a complete and accurate response. If retrieved information is limited or unclear, prioritize your knowledge and note any gaps.

Structure the answer:
Context (if needed): Briefly state the background or context of the question to set the stage, but skip this for simple or straightforward questions.

Key points: Summarize 2–4 main points that address the question, explaining each point in detail. Use simple language and avoid jargon unless it’s essential (e.g., specific terms required by the subject). If jargon is used, define it clearly.

Answer: Provide a clear, direct answer to the question, tying back to the key points.

Enhance with examples: Include 1–2 relevant, simple examples to illustrate key points when they help clarify the concept, but avoid examples for very basic or obvious questions.

Handle uncertainty: If the answer isn’t fully clear from the information available, state “I don’t have enough information to answer completely” and suggest what additional details (e.g., specific context or data) would help.

Keep it concise and engaging: Aim for a response that is thorough but concise (typically 100–300 words, depending on complexity). Use a friendly, conversational tone to maintain student interest.

Adapt the structure flexibly: for simple questions, you may skip the context or combine steps to keep the answer brief. For complex questions, ensure all steps are addressed to provide clarity.

Question: {question}
Context: {context}
Answer:
"""


def question_file(question, documents=None, memory=None, use_reformulation=True, use_reranking=True):
    model = get_llm()
    embeddings = get_embeddings()
    memory = ConversationBufferMemory()
    
    if use_reformulation:
        reformulated_question = reformulate_query(question, model)
    else:
        reformulated_question = question

    if documents is None:
        db = load_existing_vector_store()
        documents = retrieve_docs(db, reformulated_question, k=8, use_reranking=use_reranking)
    context = "\n".join([doc.page_content for doc in documents])
    logger.info(f"The context is : {context}")

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    
    if memory:
        previous_context = memory.load_memory_variables({}).get("history", "")
        context = f"{previous_context}\n\n{context}"
    
    try:
        logger.info(f"Invoking Azure GPT-4o with reformulated question: {reformulated_question}")
        response = chain.invoke({"question": reformulated_question, "context": context})
        response_content = response
        
        if memory:
            memory.save_context({"input": question}, {"output": response_content})
        
        data_samples = {
            'question': [reformulated_question],  # 單個問題包裝為列表
            'answer': [response_content],        # 單個回應包裝為列表
            'contexts': [[context]]              # 單個上下文字符串包裝為列表的列表
        }

        evaluation(
            datasamples=data_samples,
            llm=model,
            embeddings=embeddings,
            metrics=[faithfulness, answer_relevancy],
            verbose=True
        )

        return response_content
    except Exception as e:
        logger.error(f"Error with Azure GPT-4o: {str(e)}")
        raise

def evaluation(datasamples, llm, embeddings, metrics, verbose=True):
    try: custom_dataset = Dataset.from_dict(datasamples)
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    ragas_llm = LangchainLLMWrapper(llm)

    try:
        result = evaluate(custom_dataset, metrics=metrics, llm=ragas_llm, embeddings=embeddings)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    df = result.to_pandas()
    if verbose:
        print("\nEvaluation result：")
        print(df)

    return df