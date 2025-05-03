from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import logging
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, EMBEDDING_DEPLOYMENT, LLM_DEPLOYMENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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