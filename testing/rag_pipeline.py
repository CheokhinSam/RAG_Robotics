from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from langsmith import traceable
from llm import get_llm
from vector_store import retrieve_docs, load_existing_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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