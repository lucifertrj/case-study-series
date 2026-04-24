# CLAUDE.md — HOME-TAKER Backend

## What We're Building

Backend API for a caregiving assistant using **Cognee** as AI memory. Onboard → Log changes → Chat with memory.

---

## Stack

- **API:** FastAPI
- **AI Memory:** Cognee (see `SKILL.md`)
- **LLM:** OpenAI (via `LLM_API_KEY` — already configured)

No database. Cognee handles all memory persistence.

---

## Setup

```bash
pip install cognee fastapi uvicorn openai
```

Env (already present):
```
LLM_API_KEY=<your-openai-key>
```

---

## Core Flows

### 1. Onboarding → Ingest to Cognee

```python
import cognee

async def ingest_onboarding(data: dict):
    profile = f"""
    CARE PROFILE: {data['name']}
    DOB: {data['dateOfBirth']} | Blood: {data['bloodType']}
    
    EMERGENCY: {data['emergencyContact']['name']} ({data['emergencyContact']['relationship']}) - {data['emergencyContact']['phone']}
    DOCTOR: {data['doctor']['name']} - {data['doctor']['phone']}
    
    ALLERGIES: {', '.join(data['allergies'])}
    CONDITIONS: {', '.join(data['conditions'])}
    
    MEDICATIONS:
    {chr(10).join([f"- {m['name']} {m['dosage']} | {m['frequency']} at {m['time']} | Appearance: {m['color']}" for m in data['medications']])}
    """
    
    await cognee.add(profile)
    await cognee.cognify()
```

### 2. Log Entry → Add to Memory

```python
async def add_log(entry: dict):
    log_text = f"[{entry['type'].upper()}] {entry['title']}\n{entry['details']}"
    
    await cognee.add(log_text)
    await cognee.cognify()
```

### 3. Chat → Search Memory → Respond

```python
import openai

async def chat(user_message: str):
    # 1. Query Cognee
    results = await cognee.search(query_text=user_message)
    memory_context = "\n".join([r['search_result'][0] for r in results])
    
    # 2. Call OpenAI with memory context
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""You are HOME-TAKER, a caregiving assistant.
            
USER'S CARE MEMORY:
{memory_context}

Answer using this context. Be warm and concise. If info not found, offer to log it."""},
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.choices[0].message.content
```

---

## API Endpoints

| Method | Endpoint | Action |
|--------|----------|--------|
| POST | `/onboarding` | Ingest profile → `cognee.add()` + `cognee.cognify()` |
| POST | `/log` | Add memory entry → `cognee.add()` + `cognee.cognify()` |
| POST | `/chat` | Search memory → OpenAI response |

---

## Key Notes

1. **Every write calls `cognee.add()` then `cognee.cognify()`**
2. **Chat always queries Cognee first** — inject results into OpenAI system prompt
3. **Medication appearance matters** — track pill color/shape for change detection

---

## Reference

- Cognee usage: `SKILL.md`
- Cognee repo: https://github.com/topoteretes/cognee