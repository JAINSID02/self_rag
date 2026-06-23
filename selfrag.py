
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import List , TypedDict , Literal
from pydantic import BaseModel , Field

import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import  ChatPromptTemplate
from langgraph.graph import StateGraph , START , END
from langchain_ollama import OllamaEmbeddings


load_dotenv()

docs=(PyPDFLoader(r"documents\Company_Policies.pdf").load() +
      PyPDFLoader(r"documents\Company_Profile.pdf").load() +
      PyPDFLoader(r"documents\Product_and_Pricing.pdf").load())

chunks = RecursiveCharacterTextSplitter(
    chunk_size =600 , chunk_overlap=150
).split_documents(docs)

embeddings = OllamaEmbeddings(model = "nomic-embed-text")

vector_store =FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k":4})

llm =ChatOllama(model = "qwen2.5:3b")

class State(TypedDict):
    question : str
    need_retrieval:bool
    docs=List[Document]
    answer : str

class RetrieveDecision(BaseModel):
    should_retrieve : bool =Field(... , description = "True if external documents are needed to answer reliably , else False ")

decide_retrieval_prompt=ChatPromptTemplate.from_messages([("system",
            "You decide whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': boolean}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.\n"
            "- should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.\n"
            "- If unsure, choose True."
        ),("human","Question : {question}")])

shoukd_retrieve_llm = llm.with_structured_output(RetrieveDecision)

def decide_retrieval(state: State):
    decision = shoukd_retrieve_llm.invoke(
        decide_retrieval_prompt.format_messages(
            question=state["question"]
        )
    )

    print("Decision:", decision)

    return {
        "need_retrieval": decision.should_retrieve
    }
direct_generation_prompt=ChatPromptTemplate.from_messages([(
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),("human", "{question}"),])

def generate_direct(state: State):
    out =llm.invoke(direct_generation_prompt.format_messages(question=state["question"]))

    return {"answer" : out.content}

def retrieve(state : State):
    return {"docs": retriever.invoke(state["question"])}

def route_after_decide(state:State)->Literal["generate_direct" , "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    
    return "generate_direct"

g=StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)

g.add_edge(START , "decide_retrieval")
g.add_conditional_edges("decide_retrieval", route_after_decide,{"generate_direct":"generate_direct", "retrieve":'retrieve'})
g.add_edge("generate_direct", END)
g.add_edge("retrieve", END)

srag=g.compile()

result=srag.invoke(
    {
        "question": "What is Machine Learning",
        "need_retrieval": False,
        "docs": [],
        "answer": "",
    }
)

print(result["answer"])