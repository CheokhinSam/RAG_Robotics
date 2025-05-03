#Eval.py
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
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict

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



def load_existing_vector_store(collection_name="batch_default"):
    embeddings = get_embeddings()
    db = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING
    )
    logger.info(f"Loaded existing vector store for collection '{collection_name}'")
    return db

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

@traceable()
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

@traceable()
def question_file(question, documents=None, memory=None, use_reformulation=True, use_reranking=True):
    model = get_llm()
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
        
        return response_content
    except Exception as e:
        logger.error(f"Error with Azure GPT-4o: {str(e)}")
        raise
















client =Client()

# Create the dataset and examples in LangSmith
examples = [
    {
        "inputs": {"question": "What is the Concept of Multiple-Hypothesis Belief? "},
        "outputs": {"answer": "Multiple-hypothesis belief is a powerful framework for modeling uncertainty in decision-making and AI. By tracking multiple possible states or explanations, it allows systems to adapt to incomplete or noisy data. However, it requires careful trade-offs between computational efficiency and precision to remain practical in real-world applications."},
    },
    {
        "inputs": {"question": "what is Kalman filter localization?"},
        "outputs": {"answer": "The Kalman filter estimates a robot's position and state by combining predictions from motion models with updates from sensor data. It assumes Gaussian uncertainty and uses a recursive two-step process: prediction (using motion models) and update (using sensor measurements). This efficient sensor fusion method is ideal for real-time localization but may struggle with large uncertainties or multimodal distributions."},
    },
    {
        "inputs": {"question": "What is Potential field path planning?"},
        "outputs": {"answer": "Potential field path planning is a technique in robotics where an artificial field guides the robot toward a goal while avoiding obstacles. The goal acts as an attractive force, pulling the robot closer, while obstacles act as repulsive forces, pushing the robot away. The robot moves by following the gradient of this field, ensuring smooth navigation."},
    }
]

dataset_name = "Eval dataset1"
dataset = client.create_dataset(dataset_name)
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = get_llm().with_structured_output(CorrectnessGrade, method="json_schema", strict=True)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

    # Run evaluator
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions}, 
        {"role": "user", "content": answers}
    ])
    return grade["correct"]

# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

# Grade prompt
relevance_instructions="""You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = get_llm().with_structured_output(RelevanceGrade, method="json_schema", strict=True)

# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = get_llm().with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)


# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["grounded"]


# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = get_llm().with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"

    # Run evaluator
    grade = retrieval_relevance_llm.invoke(
        [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]


def main():
    # 初始化内存
    memory = ConversationBufferMemory()

    # 檢查數據庫集合
    db = load_existing_vector_store(collection_name="rag_collection")
    logger.info("Successfully loaded vector store")




    # 運行問答
    def run_question(inputs):
        question = inputs["question"]
        try:
            result = question_file(
                question=question,
                documents=None,
                memory=memory,
                use_reformulation=True,
                use_reranking=True
            )
            return {"output": result}  # 包裝為 LangSmith 預期的格式
        except Exception as e:
            logger.error(f"Failed to process question '{question}': {e}")
            return {"output": {"answer": "", "documents": []}}

    # 執行評估
    try:
        results = client.evaluate(
            run_question,
            data=dataset_name,
            evaluators=[correctness, relevance, groundedness, retrieval_relevance],
            experiment_prefix="rag-doc-evaluation",
            metadata={"version": "LCEL context, gpt-4o"}
        )
        logger.info("Evaluation completed successfully")
        print(results)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()