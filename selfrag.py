
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
    docs:List[Document]
    relevant_docs : List[Document]
    context: str
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

should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

def decide_retrieval(state: State):
    decision = should_retrieve_llm.invoke(
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




class RelevanceDecision(BaseModel):
    is_relevant : bool =Field(... , description = "True , if the documents help answer the question , else fasle" )


is_relevant_prompt =ChatPromptTemplate.from_messages([("system",
            "You are judging document relevance.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "A document is relevant if it contains information useful for answering the question."),
            ("human",
            "Question:\n{question}\n\nDocument:\n{document}")])


relevance_llm = llm.with_structured_output(RelevanceDecision)


def is_relevant(state:State):
    relevant_docs : List[Document] = []

    for doc in state["docs"]:
        decision : RelevanceDecision = relevance_llm.invoke(is_relevant_prompt.format_messages(question=state["question"],document=doc.page_content))
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs":relevant_docs}

rag_generation_prompt = ChatPromptTemplate.from_messages([(
            "system",
            "You are a business RAG assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the context does not contain enough information, say:\n"
            "'No relevant document found.'\n"
            "Do not use outside knowledge.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n"
        )])

def generate_from_context(state:State):

    context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs",[])]).strip()

    if not context:
        return {"answer": "No relevant document found.", "context": ""}
    
    out = llm.invoke(rag_generation_prompt.format_messages(question=state["question"],context=context))

    return {"answer": out.content , "context": context}

def no_relevant_docs(state:State):
    return {"answer" : "No relevant document found " , "context":""}



def route_after_decide(state:State)->Literal["generate_direct" , "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    
    return "generate_direct"


def route_after_relevance(state:State)->Literal["generate_from_context","no_relevant_docs"]:
    if state.get("relevant_docs")   and len(state["relevant_docs"]) > 0     : 
        return "generate_from_context"
    else :
        return "no_relevant_docs"


g=StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant",is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_relevant_docs", no_relevant_docs) 

g.add_edge(START , "decide_retrieval")
g.add_conditional_edges("decide_retrieval", route_after_decide,{"generate_direct":"generate_direct", "retrieve":'retrieve'})
g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")
g.add_conditional_edges("is_relevant",route_after_relevance,{"generate_from_context":"generate_from_context" , "no_relevant_docs":"no_relevant_docs"})

g.add_edge("generate_from_context", END )
g.add_edge("no_relevant_docs",END)

srag=g.compile()

result=srag.invoke(
    {
        "question": "What is the name of the book used",
        "need_retrieval": False,
        "docs": [],
        "answer": "",
    }
)

print(result["answer"])