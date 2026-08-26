import asyncio
import os
from typing import List

from utils import calculate_scores

from dotenv import load_dotenv
load_dotenv()

APP_DATASET = "rfp_offer_analyst"

async def remember_summary(
    supplier: str, tender: str, verdict_summary: str, dataset_name: str = APP_DATASET
):
    url = os.getenv("COGNEE_ENDPOINT_URL")
    api_key = os.getenv("COGNEE_API_KEY")

    import cognee

    if url and api_key:
        await cognee.serve(url=url, api_key=api_key)

    text = f"{supplier} bid on {tender}: {verdict_summary}"
    await cognee.remember(text, dataset_name=dataset_name)

    if url and api_key:
        await cognee.disconnect()


async def recall_supplier(supplier: str = "", dataset_name: str = APP_DATASET) -> List[str]:
    url = os.getenv("COGNEE_ENDPOINT_URL")
    api_key = os.getenv("COGNEE_API_KEY")

    import cognee

    if url and api_key:
        await cognee.serve(url=url, api_key=api_key)

    query = f"What do we know about {supplier}?" if supplier else "What do we know about the suppliers?"
    results = await cognee.recall(query_text=query, datasets=[dataset_name])

    memories = []
    for r in results or []:
        if isinstance(r, dict):
            memories.append(r.get("text") or r.get("raw", {}).get("value") or str(r))
        else:
            memories.append(getattr(r, "text", None) or getattr(r, "answer", None) or str(r))

    if url and api_key:
        await cognee.disconnect()

    return memories


def store_tender_memories(conn, tender_id: str = "TND-2026-0418") -> None:
    ranked, max_score = calculate_scores(conn)
    if not ranked:
        return

    for r in ranked:
        supplier = r["supplier"]
        if r["is_disqualified"]:
            summary = f"Disqualified on knockout ({r['dq_reason']}). Score: {r['score']}/{max_score}."
        elif r["unclear_count"] > 0:
            summary = f"Score: {r['score']}/{max_score} with {r['unclear_count']} unclear requirement(s)."
        else:
            summary = f"Full evaluation complete with score {r['score']}/{max_score}."

        asyncio.run(remember_summary(supplier, tender_id, summary))


_BOILERPLATE_MARKERS = (
    "no information",
    "does not make sense",
    "no entries",
    "no other entities",
    "no suppliers",
    "nothing is known",
    "no relationships",
    "cannot find",
    "no data",
    "no knowledge graph",
)


def _sanitize_memories(memories: List[str], known_suppliers: set) -> List[str]:
    cleaned: List[str] = []
    seen = set()

    for raw in memories:
        text = " ".join((raw or "").split())
        if not text:
            continue

        lower = text.lower()
        if any(marker in lower for marker in _BOILERPLATE_MARKERS):
            continue
        if known_suppliers and not any(s.lower() in lower for s in known_suppliers):
            continue  # off-topic entity unrelated to this tender's actual suppliers

        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        if text in seen:
            continue

        seen.add(text)
        cleaned.append(text)

    return cleaned


def get_tender_memory_notes(conn=None, supplier_name: str = "") -> List[str]:
    ranked, max_score = calculate_scores(conn) if conn is not None else ([], 0)
    known_suppliers = {r["supplier"] for r in ranked}
    if supplier_name:
        known_suppliers = {s for s in known_suppliers if supplier_name.lower() in s.lower()}

    if known_suppliers:
        cognee_memories = asyncio.run(recall_supplier(supplier_name))
        notes = _sanitize_memories(cognee_memories, known_suppliers)
        if notes:
            return notes

    if conn is None:
        return []

    notes = []
    seen_suppliers = set()

    for r in ranked:
        supp = r["supplier"]
        if supp in seen_suppliers:
            continue
        if supplier_name and supplier_name.lower() not in supp.lower():
            continue

        seen_suppliers.add(supp)
        if r["is_disqualified"]:
            notes.append(f"**{supp}** — Disqualified on {r['dq_reason']}. Score: {r['score']}/{max_score}.")
        elif r["unclear_count"] > 0:
            notes.append(
                f"**{supp}** — {r['unclear_count']} requirement(s) unstated/unclear. Score: {r['score']}/{max_score}."
            )
        else:
            notes.append(f"**{supp}** — All requirements evaluated. Score: {r['score']}/{max_score}.")

    if not notes:
        return ["No evaluation history recorded yet. Run analysis to store supplier memory."]

    return notes
