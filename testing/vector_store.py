from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
import logging
from config import CONNECTION_STRING, WORKING_DIR
from llm import get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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