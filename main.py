# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def Home():
#     return {"Message":"Hello welcome to fast api"}

# @app.get("/about")
# def about():
#     return {"Project":"This is Loan project", "version":"1.2"}

import os

from typing import Literal

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pinecone import Pinecone

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_pinecone import (
    PineconeVectorStore,
    PineconeEmbeddings
)

from langchain_groq import ChatGroq

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import MessagesState

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class RAGState(TypedDict):
    question: str
    context: str
    retry_count: int
    answer: str

load_dotenv()
app = FastAPI()


INDEX_NAME = os.environ["PINECONE_INDEX"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing API keys in .env file")

# =====================================
# FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Medical API",
    description="RAG application using LangGraph, Groq and Pinecone",
    version="1.0.0"
)

# =====================================
# LLM INITIALIZATION
# =====================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

# =====================================
# PINECONE INITIALIZATION
# =====================================

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2"
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# =====================================
# PYDANTIC MODELS
# =====================================

class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description="yes or no"
    )


class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=1,
        description="User question"
    )


class AnswerResponse(BaseModel):

    question: str
    answer: str


# =====================================
# PROMPTS
# =====================================

GRADE_PROMPT = """
You are a document relevance grader.

Question:
{question}

Retrieved Content:
{context}

If the retrieved content is relevant
to the question respond with: yes
otherwise: no
"""

ANSWER_PROMPT = """
You are a helpful assistant.

Use ONLY the provided context.

Question:
{question}

Context:
{context}

Answer:
"""

REWRITE_PROMPT = """
Rewrite the user question to improve retrieval.

Question:
{question}
"""

# =====================================
# NODE 1: RETRIEVE DOCUMENTS
# =====================================

def generate_query_or_respond(state: RAGState):

    question = state["question"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {
        "context": context
    }

# =====================================
# NODE 2: GRADE DOCUMENTS
# =====================================

def grade_documents(state: RAGState):

    question = state["question"]
    context = state["context"]

    if not context.strip():
        if state["retry_count"] >= 2:
            return "fallback"

        return "rewrite_question"

    prompt = GRADE_PROMPT.format(
        question=question,
        context=context
    )

    result = llm.with_structured_output(
        GradeDocuments,
        method="json_schema",
        strict=True
    ).invoke(prompt)

    if result.binary_score.lower() == "yes":
        return "generate_answer"

    if state["retry_count"] >= 2:
        return "fallback"

    return "rewrite_question"


# =====================================
# NODE 3: REWRITE QUESTION
# =====================================

def rewrite_question(state: RAGState):

    question = state["question"]

    rewritten = llm.invoke(
        REWRITE_PROMPT.format(
            question=question
        )
    )

    return {
        "question": rewritten.content,
        "retry_count": state["retry_count"] + 1
    }



# =====================================
# NODE 4: GENERATE ANSWER
# =====================================

def generate_answer(state: RAGState):

    question = state["question"]
    context = state["context"]

    prompt = ANSWER_PROMPT.format(
        question=question,
        context=context
    )

    answer = llm.invoke(prompt)

    return {
        "answer": answer.content
    }


def fallback(state: RAGState):

    return {
        "answer": (
            "Sorry, I could not find relevant information "
            "in the available documents."
        )
    }


# =====================================
# LANGGRAPH WORKFLOW
# =====================================

workflow = StateGraph(RAGState)

workflow.add_node(
    "retrieve",
    generate_query_or_respond
)

workflow.add_node(
    "generate_answer",
    generate_answer
)

workflow.add_node(
    "rewrite_question",
    rewrite_question
)

workflow.add_node(
    "fallback",
    fallback
)

workflow.add_edge(START, "retrieve")

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
        "fallback": "fallback"
    }
)

workflow.add_edge(
    "rewrite_question",
    "retrieve"
)

workflow.add_edge("generate_answer", END)

workflow.add_edge("fallback", END)

graph = workflow.compile()

# =====================================
# ASK QUESTION FUNCTION
# =====================================

def ask_question(question: str):

    result = graph.invoke(
        {
            "question": question,
            "context": "",
            "retry_count": 0,
            "answer": ""
        },
        config={
            "recursion_limit": 15
        }
    )

    return result["answer"]

# =====================================
# FASTAPI ROUTES
# =====================================

@app.get("/")
def home():

    return {
        "message": "LangGraph RAG API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: QuestionRequest):

    try:

        answer = ask_question(
            request.question
        )

        return AnswerResponse(
            question=request.question,
            answer=answer
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    