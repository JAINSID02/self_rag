
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

    is_sup:Literal["Fully Supported ","Partially Supported", "No support"]
    evidence:List[str]
    retries : int

    is_use :Literal["useful","not_useful"]
    use_reason:str

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

def route_after_decide(state:State)->Literal["generate_direct" , "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    
    return "generate_direct"

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

def route_after_relevance(state:State)->Literal["generate_from_context","no_answer_found"]:
    if state.get("relevant_docs")   and len(state["relevant_docs"]) > 0     : 
        return "generate_from_context"
    else :
        return "no_answer_found"


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

def no_answer_found(state:State):
    return {"answer" : "No answer found " , "context":""}


class is_SupDecision(BaseModel):
    issup:Literal["Fully_Supported" , "partially_supported" , "no_support"]
    evidence:List[str]=Field(default_factory=list)


is_sup_prompt = ChatPromptTemplate.from_messages([(
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge."
        ),(
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
        )])

is_sup_llm = llm.with_structured_output(is_SupDecision)

def is_sup(state:State):
    decision : is_SupDecision = is_sup_llm.invoke(is_sup_prompt.format_messages(
        question=state["question"],
        answer=state.get("answer",""),
        context=state.get("context","")
    ))

    return {"issup":decision.issup , "evidence":decision.evidence}

MAX_RETRIES = 10

def route_after_is_sup(state:State)->Literal["accept_answer","revise_answer"]:
    if state.get("is_sup")=="Fully Supported":
        return "accept_answer"
    
    if state.get("retries", 0) >=MAX_RETRIES:
        return "accept_answer"
    
    return "revise_answer"

def accept_answer(state:State):
    return{}

revise_prompt = ChatPromptTemplate.from_messages([(
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
        ),(
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        )])

def revise_answer(state:State):
    out=llm.invoke(revise_prompt.format_messages(
        question=state["question"],
        answer=state.get("answer",""),
        context= state.get("context","")
    ))

    return {"answer":out.content,
            "retries": state.get("retries" , 0) +1}


class is_use_decision(BaseModel):
    is_use:Literal["useful","not_useful"]
    reason:str=Field(... , description = "short answer in one line")

is_use_prompt=ChatPromptTemplate.from_messages([(
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line."
        ),(
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        )])

is_use_llm=llm.with_structured_output(is_use_decision)

def is_use(state:State):
    decision:is_use_decision=is_use_llm.invoke(is_use_prompt.format_messages(
        question = state["question"],
        answer=state.get("answer","")
    ))

    return {"is_use":decision.is_use , "use_reason":decision.reason}

def route_after_is_use(state:State)->Literal["END","no_answer_found"]:
    if state.get("is_use")=="useful":
        return "END"
    return "no_answer_found"







g=StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant",is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_answer_found", no_answer_found) 
g.add_node("is_sup",is_sup)
g.add_node("revise_answer",revise_answer)
g.add_node("is_use",is_use)





g.add_edge(START , "decide_retrieval")
g.add_conditional_edges("decide_retrieval", route_after_decide,{"generate_direct":"generate_direct", "retrieve":'retrieve'})
g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")
g.add_conditional_edges("is_relevant",route_after_relevance,{"generate_from_context":"generate_from_context" , "no_answer_found":"no_answer_found"})
g.add_edge("no_answer_found",END)
g.add_edge("generate_from_context","is_sup")
g.add_conditional_edges("is_sup",route_after_is_sup,{"accept_answer":"is_use","revise_answer":"revise_answer"})
g.add_edge("revise_answer","is_sup")
g.add_conditional_edges("is_use",route_after_is_use,{"END":END,"no_answer_found":"no_answer_found"})

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