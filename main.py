"""
This script implements a Retrieval-Augmented Generation (RAG) chatbot.
The chatbot uses a combination of semantic and keyword search to retrieve relevant documents
from a medical Q&A dataset. The retrieved documents are then re-ranked and used to generate
an answer to a user's question. The entire process is orchestrated using a state graph.
"""
import os
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from typing import List, TypedDict
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from flashrank import Ranker, RerankRequest
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Environment Setup ---
# Make sure you have Ollama running with the llama3:8b model pulled
# IMPORTANT: You must install the following packages:
# pip install -U langchain-chroma
# pip install rank_bm25
# pip install flashrank

# --- 1. Document Loading and Processing ---
csv_file_path = './MedicalQ&A.csv'
persist_directory = './chroma_db'

# --- 2. Vector Store and Retrievers ---
embeddings = OllamaEmbeddings(model="llama3:8b")

# Check if the vector store already exists
if os.path.exists(persist_directory):
    print("---LOADING EMBEDDINGS FROM DISK---")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("---COMPUTING AND STORING EMBEDDINGS---")
    loader = CSVLoader(file_path=csv_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )

# Load all splits for the BM25 retriever (it works in-memory)
loader = CSVLoader(file_path=csv_file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize Semantic Search Retriever
vector_retriever = vectorstore.as_retriever()

# Initialize Keyword Search Retriever
bm25_retriever = BM25Retriever.from_documents(splits)

# Initialize Ensemble Retriever for Hybrid Search
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)

# --- 3. Re-ranker Setup ---
# Initialize the FlashRank re-ranker
# The first time this runs, it will download the model automatically
ranker = Ranker()

# --- 4. Language Model ---
llm_json = ChatOllama(model="llama3:8b", format="json", temperature=0)
llm_string = ChatOllama(model="llama3:8b", temperature=0)


# --- 5. Graph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question (str): The user's question.
        generation (str): The LLM's generated answer.
        documents (List[str]): A list of retrieved documents.
        retries (int): The number of times the query has been transformed.
    """
    question: str
    generation: str
    documents: List[str]
    retries: int

# --- 6. Graph Nodes ---

def retrieve(state):
    """
    Retrieves documents from the vector store using a hybrid search approach.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: A dictionary with the updated documents, question, and retries.
    """
    print("---RETRIEVING DOCUMENTS (HYBRID SEARCH)---")
    question = state["question"]
    # Invoke the ensemble retriever to get documents using both semantic and keyword search
    documents = ensemble_retriever.invoke(question)
    return {"documents": documents, "question": question, "retries": state.get("retries", 0)}

def rerank(state):
    """
    Re-ranks the retrieved documents using a local cross-encoder model to improve relevance.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: A dictionary with the re-ranked documents.
    """
    print("---RE-RANKING DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    # Format the documents for the FlashRank re-ranker
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
    
    # Create a rerank request with the user's query and the retrieved passages
    rerank_request = RerankRequest(query=question, passages=passages)
    
    # Perform the re-ranking
    reranked_results = ranker.rerank(rerank_request)
    
    # Reorder the original Document objects based on the new ranking
    reranked_docs = [documents[res["id"]] for res in reranked_results]
    
    return {"documents": reranked_docs}


def generate(state):
    """
    Generates an answer using the retrieved and re-ranked documents.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: A dictionary with the updated documents, question, and the generated answer.
    """
    print("---GENERATING ANSWER FROM RAG---")
    question = state["question"]
    documents = state["documents"]

    # Define the prompt for the RAG chain
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:""",
        input_variables=["question", "context"],
    )

    # Create the RAG chain by piping the prompt, LLM, and output parser
    rag_chain = prompt | llm_string | StrOutputParser()
    # Invoke the chain with the context and question to generate an answer
    generation = rag_chain.invoke({"context": documents, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the user's question.
    Filters out documents that are not relevant.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: A dictionary with the filtered documents, question, and retries.
    """
    print("---CHECKING DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    # Define the prompt for the document grader
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        Here is the retrieved document:
        \n\n {document} \n\n
        Here is the user question: {question}
        If the document contains keywords related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["question", "document"],
    )

    # Create a chain for grading documents
    chain = prompt | llm_json | JsonOutputParser()
    
    filtered_docs = []
    # Iterate through the documents and grade each one
    for d in documents:
        score = chain.invoke({"question": question, "document": d.page_content})
        grade = score.get('score', 'no')
        if grade.lower() == "yes":
            # print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            # print("---GRADE: DOCUMENT NOT RELEVANT---")
            pass
    
    return {"documents": filtered_docs, "question": question, "retries": state.get("retries", 0)}


def transform_query(state):
    """
    Transforms the user's query to produce a better question for retrieval.
    This is used when the initial retrieval results are not relevant.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: A dictionary with the new, improved question, the documents, and an incremented retry count.
    """
    print("---TRANSFORMING QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Define the prompt for query transformation
    prompt = PromptTemplate(
        template="""You are generating questions that are well optimized for retrieval.
        Look at the input and try to reason about the underlying semantic intent / meaning.
        Here is the initial question:
        \n ------- \n
        {question}

        \n ------- \n
        Formulate an improved question as a JSON with a single key 'improved_question' and no preamble or explanation.""",
        input_variables=["question"],
    )

    # Create a chain for transforming the query
    chain = prompt | llm_json | JsonOutputParser()
    # Invoke the chain to get an improved question
    new_question_json = chain.invoke({"question": question})
    new_question = new_question_json.get("improved_question", question)
    
    return {"question": new_question, "documents": documents, "retries": state.get("retries", 0) + 1}

def fallback(state):
    """
    Generates a response directly from the LLM without using RAG context.
    This is a fallback mechanism when the RAG pipeline fails to find relevant documents.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: A dictionary with the generated answer.
    """
    print("---FALLING BACK TO LLM---")
    question = state["question"]
    # Define a prompt for the fallback scenario
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Answer the following question to the best of your ability as the RAG pipeline failed to find relevant documents.
        Question: {question}
        Answer:""",
        input_variables=["question"],
    )
    # Create a chain to generate an answer directly from the LLM
    chain = prompt | llm_string | StrOutputParser()
    generation = chain.invoke({"question": question})
    return {"generation": generation}

def decide_to_generate(state):
    """
    Determines the next step in the graph based on the relevance of the graded documents.
    It can decide to generate an answer, transform the query, or fallback to the LLM.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        str: A string indicating the next node to execute ('generate', 'transform_query', or 'fallback').
    """
    print("---ASSESSING GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    retries = state.get("retries", 
    0)

    if not filtered_documents:
        # If no documents are relevant
        if retries < 1:
            # If we haven't retried yet, transform the query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, TRANSFORMING QUERY---")
            return "transform_query"
        else:
            # If we have already retried, fallback to the LLM
            print("---DECISION: FAILED AFTER RETRY, FALLING BACK TO LLM---")
            return "fallback"
    else:
        # If there are relevant documents, proceed to generate an answer
        print("---DECISION: GENERATE---")
        return "generate"

# --- 7. Graph Definition and Execution ---
# This section defines the structure of the state graph and compiles it into a runnable application.

# Initialize the state graph with the defined GraphState
workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("fallback", fallback)

# Define the edges and control flow of the graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade_documents")
# Add conditional edges based on the output of the 'decide_to_generate' function
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "fallback": "fallback",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

# Compile the graph into a runnable application
app = workflow.compile()

# --- 8. Save the RAG pipeline as a PNG ---
def save_graph_image(app, filename="workflow.png"):
    """
    Saves a visual representation of the graph to a PNG file.

    Args:
        app: The compiled LangGraph application.
        filename (str): The name of the file to save the image to.
    """
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_bytes)
        print(f"---SAVED GRAPH IMAGE TO {filename}---")
    except Exception as e:
        print(f"---ERROR SAVING GRAPH IMAGE: {e}---")
        print("Please make sure you have graphviz installed: pip install graphviz")

# --- 9. Run the RAG pipeline ---
# This section provides a function to run the chatbot and an interactive loop for user input.

def run_chatbot(question: str):
    """
    Runs the RAG pipeline with a given question and prints the final answer.

    Args:
        question (str): The user's question.
    """
    # Set the initial state for the graph
    inputs = {"question": question, "retries": 0}
    # Stream the outputs of the graph execution
    for output in app.stream(inputs):
        for key, value in output.items():
            # Print the final answer when it's generated
            if key == "generate" or key == "fallback":
                print(f"\nAnswer: {value['generation']}")
    print("\n---\n")


# --- Interactive Chat Loop ---
if __name__ == "__main__":
    # Save the graph visualization
    save_graph_image(app)
    
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            break
        run_chatbot(user_question)
