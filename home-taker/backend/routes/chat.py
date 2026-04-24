from fastapi import APIRouter
from openai import OpenAI

from backend.config import LLM_API_KEY, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from backend.models import ChatRequest, ChatResponse
from backend.services import CogneeService

router = APIRouter(prefix="/api", tags=["chat"])

# Initialize OpenAI client
openai_client = OpenAI(api_key=LLM_API_KEY)

SYSTEM_PROMPT_TEMPLATE = """You are HOME-TAKER, a warm and caring assistant for caregivers and elderly users.
You help them manage health information, medications, appointments, and daily care.

USER'S CARE MEMORY:
{memory_context}

Guidelines:
- Answer using the memory context above when relevant
- Be warm, concise, and helpful
- If information isn't in memory, offer to help log it
- For medication questions, always mention appearance if available (important for safety)
- For emergency information, be clear and direct
- Keep responses conversational but informative"""


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Search Cognee memory and respond using OpenAI."""
    try:
        memory_context = await CogneeService.search(request.message)

        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_TEMPLATE.format(memory_context=memory_context),
                },
                {"role": "user", "content": request.message},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        return ChatResponse(response=response.choices[0].message.content)

    except Exception as e:
        print(f"Chat error: {e}")
        return ChatResponse(
            response="I'm having trouble accessing my memory right now. Please make sure your profile is set up and try again."
        )
