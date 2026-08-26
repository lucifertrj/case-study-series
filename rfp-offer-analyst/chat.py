import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from client import get_llm_client
from config import config
from vector_store import search_offer


class ChatState(TypedDict):
    offer_id: int
    supplier_name: str
    query: str
    pages: List[List[Any]]
    hits: List[Dict[str, Any]]
    answer: str
    citations: List[str]
    error: str


def retrieve_offer_context(state: ChatState) -> Dict[str, Any]:
    try:
        hits = search_offer(state["offer_id"], state["query"])
        return {"hits": hits, "error": ""}
    except Exception as err:
        return {"hits": [], "error": f"Vector search unavailable: {err}"}


def answer_from_context(state: ChatState) -> Dict[str, Any]:
    if state["error"]:
        return {"answer": state["error"], "citations": []}

    hits = state["hits"]
    if not hits:
        return {
            "answer": "I could not find relevant offer text for that question.",
            "citations": [],
        }

    citations = [f"p. {h['page']}" for h in hits[:2]]
    context_snippets = "\n".join(f"[p. {h['page']}] {h['text']}" for h in hits)
    gclient = get_llm_client(required=False)

    if not gclient:
        return {
            "answer": context_snippets,
            "citations": citations,
        }

    try:
        from google.genai import types

        prompt = (
            f"You are assisting a buyer analyzing the offer from {state['supplier_name']}.\n"
            "Answer the user question based solely on the provided Qdrant-retrieved offer text.\n"
            "Be concise, clear, and include page references when relevant.\n\n"
            f"Offer Context:\n{context_snippets}\n\n"
            f"User Question: {state['query']}\n"
        )
        resp = gclient.models.generate_content(
            model=config.llm.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=config.llm.chat_max_output_tokens,
                temperature=config.llm.chat_temperature,
            ),
        )
        return {"answer": (resp.text or "").strip(), "citations": citations}
    except Exception as err:
        return {"answer": f"LLM answer unavailable: {err}", "citations": citations}


chat_builder = StateGraph(ChatState)
chat_builder.add_node("retrieve", retrieve_offer_context)
chat_builder.add_node("answer", answer_from_context)
chat_builder.add_edge(START, "retrieve")
chat_builder.add_edge("retrieve", "answer")
chat_builder.add_edge("answer", END)

chat_graph = chat_builder.compile()


def answer_offer_question(offer_id: int, supplier_name: str, query: str, pages_json: str) -> Dict[str, Any]:
    output = chat_graph.invoke(
        {
            "offer_id": offer_id,
            "supplier_name": supplier_name,
            "query": query,
            "pages": json.loads(pages_json),
            "hits": [],
            "answer": "",
            "citations": [],
            "error": "",
        }
    )
    return {
        "answer": output.get("answer", ""),
        "citations": output.get("citations", []),
    }
