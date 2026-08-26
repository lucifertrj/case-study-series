import json
import operator
import re
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from client import get_llm_client
from config import config

def find_page_number(quote: str, pages: List[List[Any]]) -> int:
    if not quote or not pages:
        return 1

    clean_quote = re.sub(r"\s+", " ", quote).strip().lower()
    if not clean_quote:
        return 1

    for page_num, text in pages:
        clean_text = re.sub(r"\s+", " ", str(text)).strip().lower()
        if clean_quote in clean_text:
            return int(page_num)

    words = clean_quote.split()
    if len(words) >= 3:
        snippet = " ".join(words[:5])
        for page_num, text in pages:
            clean_text = re.sub(r"\s+", " ", str(text)).strip().lower()
            if snippet in clean_text:
                return int(page_num)

    return 1


class GradingError(RuntimeError):
    pass


def llm_grade_cell(rfp_text: str, offer_text: str, pages: List[List[Any]], criterion_q: str) -> Dict[str, Any]:
    from google.genai import types

    client = get_llm_client(required=True)
    system_prompt = (
        "You evaluate one supplier offer against one requirement.\n"
        "Answer only about the requirement given. Ignore everything else.\n"
        "If the offer does not address the requirement, answer \"unclear\" —\n"
        "do not guess, and do not treat silence as refusal.\n"
        "Return valid JSON strictly with keys: verdict (full/partial/none/unclear), "
        "reason (one concise sentence), quote (<=25 words, verbatim quote from offer or empty string), confidence (high/medium/low)."
    )

    user_prompt = (
        f"=== REQUIREMENT ===\n{criterion_q}\n\n"
        f"=== WHAT WE ASKED FOR ===\n{rfp_text}\n\n"
        f"=== THE OFFER ===\n{offer_text}\n\n"
        f"Return JSON strictly now:"
    )

    response = client.models.generate_content(
        model=config.llm.model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config.llm.grading_temperature,
            max_output_tokens=config.llm.grading_max_output_tokens,
            response_mime_type="application/json",
        ),
    )

    txt = (response.text or "").strip()
    json_match = re.search(r"\{.*\}", txt, re.DOTALL)
    if not json_match:
        raise GradingError(f"Gemini returned no parseable JSON: {txt[:200]!r}")

    data = json.loads(json_match.group(0))
    verdict = str(data.get("verdict", "")).lower()
    if verdict not in {"full", "partial", "none", "unclear"}:
        raise GradingError(f"Gemini returned an invalid verdict: {data.get('verdict')!r}")

    quote = data.get("quote", "")
    return {
        "verdict": verdict,
        "reason": data.get("reason", "Evaluated against requirement."),
        "quote": quote,
        "page": find_page_number(quote, pages),
        "confidence": data.get("confidence", "high"),
    }


class CellState(TypedDict):
    offer_id: int
    offer_name: str
    supplier: str
    pages_json: str
    criterion_id: int
    question: str
    rfp_text: str

class OverallState(TypedDict):
    rfp_text: str
    offers: List[Dict[str, Any]]
    criteria: List[Dict[str, Any]]
    db_conn: Any
    results: Annotated[List[Dict[str, Any]], operator.add]


def plan_grading(state: OverallState) -> List[Send]:
    sends = []
    for offer in state["offers"]:
        for crit in state["criteria"]:
            sends.append(
                Send(
                    "grade_cell",
                    {
                        "offer_id": offer["id"],
                        "offer_name": offer["name"],
                        "supplier": offer["supplier"],
                        "pages_json": offer["pages_json"],
                        "criterion_id": crit["id"],
                        "question": crit["question"],
                        "rfp_text": state["rfp_text"],
                    },
                )
            )
    return sends


def grade_cell_node(state: CellState) -> Dict[str, Any]:
    try:
        pages = json.loads(state["pages_json"])
        offer_full_text = "\n\n".join(txt for _, txt in pages)

        res = llm_grade_cell(
            rfp_text=state["rfp_text"],
            offer_text=offer_full_text,
            pages=pages,
            criterion_q=state["question"],
        )

        row = {
            "offer_id": state["offer_id"],
            "criterion_id": state["criterion_id"],
            "verdict": res["verdict"],
            "reason": res["reason"],
            "quote": res["quote"],
            "page": res["page"],
            "confidence": res["confidence"],
        }
        return {"results": [row]}
    except Exception as err:
        return {
            "results": [{
                "offer_id": state["offer_id"],
                "criterion_id": state["criterion_id"],
                "verdict": "unclear",
                "reason": f"Evaluation error: {str(err)}",
                "quote": "",
                "page": 1,
                "confidence": "low",
            }]
        }


def collect_results(state: OverallState) -> OverallState:
    return state


builder = StateGraph(OverallState)
builder.add_node("grade_cell", grade_cell_node)
builder.add_node("collect", collect_results)

builder.add_conditional_edges(START, plan_grading, ["grade_cell"])
builder.add_edge("grade_cell", "collect")
builder.add_edge("collect", END)

graph = builder.compile()


def run_grading_batch(conn, rfp_text: str, progress_callback=None):
    offers = [dict(r) for r in conn.execute("SELECT * FROM offers ORDER BY id").fetchall()]
    criteria = [dict(r) for r in conn.execute("SELECT * FROM criteria ORDER BY id").fetchall()]

    if not offers or not criteria:
        return

    conn.execute("DELETE FROM evaluations")
    conn.commit()

    initial_state = {
        "rfp_text": rfp_text or "Standard RFP requirements",
        "offers": offers,
        "criteria": criteria,
        "db_conn": conn,
        "results": [],
    }

    output = graph.invoke(initial_state, {"max_concurrency": 20})

    for r in output.get("results", []):
        conn.execute(
            """
            INSERT OR REPLACE INTO evaluations
            (offer_id, criterion_id, verdict, reason, quote, page, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["offer_id"],
                r["criterion_id"],
                r["verdict"],
                r["reason"],
                r["quote"],
                r["page"],
                r["confidence"],
            ),
        )
    conn.commit()
