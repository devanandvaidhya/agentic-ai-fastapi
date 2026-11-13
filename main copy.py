import os
from typing import Literal

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from pydantic import BaseModel, Field
from langchain_pinecone import PineconeEmbeddings
from langchain_groq import ChatGroq


load_dotenv()

INDEX_NAME = os.environ["PINECONE_INDEX"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0
# )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2"
)

pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# GRADER
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="yes or no"
    )

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

# Node-1
def generate_query_or_respond(state: MessagesState):
    questions = state["messages"][-1].content
    docs = retriever.invoke(questions)

    context = "\n\n".join(doc.page_content
                          for doc in docs)

    return {
        "messages":[HumanMessage(content=context)]
    }


# Node-2
def grade_documents(state:MessagesState)-> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content

    promt = GRADE_PROMPT.format(
        question=question,
        context=context
    )

    # result = (llm.with_structured_output(GradeDocuments).invoke(promt))

    result = llm.with_structured_output(
    GradeDocuments,
    method="json_schema",
    strict=True
).invoke(promt)
    

    if result.binary_score.lower()=="yes":
        return "generate_answer"
    return "rewrite_question"

# Node-3
def rewrite_question(state:MessagesState):
    question = state["messages"][0].content
    rewritten = llm.invoke(REWRITE_PROMPT.format(question=question))

    return {"messages":[HumanMessage(content=rewritten.content)]}

# Node -4
def generate_answer(state:MessagesState):
    question=state["messages"][0].content
    context =state["messages"][-1].content

    prompt = ANSWER_PROMPT.format(question=question,context=context)

    answer = llm.invoke(prompt)
    return {"messages":[answer]}

#GRAPH

workflow = StateGraph(MessagesState)
workflow.add_node("retrieve",generate_query_or_respond)
workflow.add_node("generate_answer",generate_answer)
workflow.add_node("rewrite_question",rewrite_question)
workflow.add_edge(START,"retrieve")
workflow.add_conditional_edges("retrieve",grade_documents)
workflow.add_edge("rewrite_question","retrieve")
workflow.add_edge("generate_answer",END)

graph = workflow.compile()

def ask_question(question:str):
    result = graph.invoke({"messages":[HumanMessage(content=question)]})

    print("\n\n ==========FINAL STATE==========")
    for i, msg in enumerate(result["messages"]):
        print(f"\nMessage {i+1}")
        print(f"Type: {type(msg).__name__}")
        print(msg.content)
        print("-" * 80)
        
    return result["messages"][-1].content


if __name__ == "__main__":

    while True:

        q = input("User: ")

        if q.lower() in ["exit", "quit"]:
            break

        answer = ask_question(q)

        print("\nAssistant:")
        print(answer)
