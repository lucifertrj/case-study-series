import streamlit as st
import os
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_sambanova import ChatSambaNovaCloud
from langchain.schema import HumanMessage, SystemMessage

st.set_page_config(
    page_title="Uber Case Study Prototype", 
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    h1 {
        text-align: center;
        color: #1f1f1f;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
COLLECTION_NAME = "casestudy"

@st.cache_resource
def init_models():
    dense_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-small-en")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/BM25")
    
    client = QdrantClient(
        url=st.secrets['QDRANT_URL'],
        api_key=st.secrets['QDRANT_API_KEY'],
    )
    
    os.environ['SAMBANOVA_API_KEY'] = st.secrets["SAMBANOVA_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
    
    small_llm = ChatSambaNovaCloud(
        model="Meta-Llama-3.1-8B-Instruct",
        max_tokens=4096,
        temperature=0.0
    )
    
    large_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        max_tokens=None,
        temperature=0
    )
    
    return dense_model, sparse_model, client, small_llm, large_llm

class AgenticRAGState(TypedDict):
    query: str
    context: List[Dict[str, Any]]
    answer: str
    filter_conditions: Optional[Dict[str, Any]]

dense_model, sparse_model, client, small_llm, large_llm = init_models()

def rewriter_agent(state: AgenticRAGState) -> AgenticRAGState:
    """Query optimization using small LLM"""
    query = state["query"]
    
    prompt = f"""Improve this query for better search results:
    Query: {query}

    Just rewrite it to be more specific and searchable. Keep it simple.
    Return only the improved query, nothing else.
    """
    
    response = small_llm.invoke([HumanMessage(content=prompt)])
    state["query"] = response.content.strip()
    
    return state

def search_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """Hybrid search - Retrieval process"""
    query = state["query"]
    filter_condition = state["filter_conditions"]
    
    dense_vectors = next(dense_model.embed([query]))
    sparse_vectors = next(sparse_model.embed([query]))
    
    prefetch = [
        models.Prefetch(query=dense_vectors, using="dense", limit=6),
        models.Prefetch(query=models.SparseVector(**sparse_vectors.as_object()), using="sparse", limit=6)
    ]
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=dense_vectors,
        using="dense",
        query_filter=filter_condition,
        with_payload=True,
        limit=4,
    )
    
    context = []
    for result in results.points:
        context.append({
            "content": result.payload["content"],
            "source": result.payload["source"],
            "page": result.payload.get("page", 0),
            "score": result.score,
            "chunk_keywords": result.payload.get('chunk_keywords', []),
        })
    
    state["context"] = context
    return state

def answer_generation(state: AgenticRAGState) -> AgenticRAGState:
    """Generate answer using large LLM"""
    query = state["query"]
    context = state["context"]
    
    context_text = ""
    for chunk in context:
        context_text += f"{chunk['source']} (Page {chunk['page']}): {chunk['content']}\n\n"
    
    prompt = f"""Answer this question using the policy context provided:
    Question: {query}
    Context:
    {context_text}
    
    Instructions:
    1. If the message is a greeting (hello, hi, good morning, how are you, etc.), respond warmly and ask how you can help with policy-related questions or information from the available data.
    2. If the question can be answered using the context provided, give a clear, accurate answer citing the relevant sources.
    3. If the question is NOT a greeting AND cannot be answered from the context, respond with: "Not enough information available. Improve your query with relevant keywords."
    
    Be conversational and helpful while staying focused on the available policy information.
    """

    response = large_llm.invoke([HumanMessage(content=prompt)])
    state["answer"] = response.content
    
    return state

@st.cache_resource
def build_workflow():
    workflow = StateGraph(AgenticRAGState)
    workflow.add_node("rewriter_agent", rewriter_agent)
    workflow.add_node("search_retrieval", search_retrieval)
    workflow.add_node("answer_generation", answer_generation)
    
    workflow.add_edge(START, "rewriter_agent")
    workflow.add_edge("rewriter_agent", "search_retrieval")
    workflow.add_edge("search_retrieval", "answer_generation")
    workflow.add_edge("answer_generation", END)
    
    return workflow.compile()

graph = build_workflow()

st.title("🚗 Uber Case Study Prototype")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if prompt := st.chat_input("Ask about policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke({
                    "query": prompt, 
                    "filter_conditions": None
                })
                
                response = result["answer"]
                st.write(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if st.session_state.messages:
    if st.button("Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()