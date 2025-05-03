from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict
from llm import get_llm

client = Client()

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

# Create the dataset and examples in LangSmith
dataset_name = "The mobile dataset2"
dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)

# Grade output schema
class CorrectnessGrade(TypedDict):
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

experiment_results = client.evaluate(
    question_file,
    data=dataset_name,
    evaluators=[correctness, relevance],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
)