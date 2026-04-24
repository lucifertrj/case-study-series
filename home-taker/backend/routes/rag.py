import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile

from backend.config import LLM_API_KEY, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from backend.models import FileUploadResponse, RAGChatRequest, RAGChatResponse
from backend.services.document_service import DocumentProcessingService
from backend.services.qdrant_service import QdrantService

router = APIRouter(prefix="/api/rag", tags=["rag"])

# Initialize OpenAI client
from openai import OpenAI

openai_client = OpenAI(api_key=LLM_API_KEY)

RAG_SYSTEM_PROMPT_TEMPLATE = """You are HOME-TAKER, a warm and caring assistant for caregivers and elderly users.
You help them manage health information, medications, appointments, and daily care.

DOCUMENT CONTEXT (from uploaded files):
{context}

Guidelines:
- Answer using the document context above when relevant
- Cite information from the documents when available
- Be warm, concise, and helpful
- If information isn't in the documents, say so clearly
- For medication questions, always mention appearance if available (important for safety)
- For emergency information, be clear and direct
- Keep responses conversational but informative"""


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile):
    """
    Upload a file, extract text, chunk it, and store embeddings in Qdrant.

    Supports: PDF, TXT, MD files
    """
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Get file extension
        file_extension = Path(file.filename or "").suffix.lower()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        try:
            # Process file: extract text and chunk
            chunks = await DocumentProcessingService.process_file(
                tmp_path, file_extension or "text"
            )

            if not chunks:
                return FileUploadResponse(
                    success=False,
                    message="No text content could be extracted from the file",
                    file_id=file_id,
                    chunks_count=0,
                )

            # Add chunks to Qdrant with embeddings
            await QdrantService.add_documents(chunks, file_id=file_id)

            return FileUploadResponse(
                success=True,
                message=f"Successfully processed {len(chunks)} chunks from {file.filename}",
                file_id=file_id,
                chunks_count=len(chunks),
            )

        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"Upload error: {e}")
        return FileUploadResponse(
            success=False,
            message=f"Error processing file: {str(e)}",
            file_id="",
            chunks_count=0,
        )


@router.post("/chat", response_model=RAGChatResponse)
async def chat(request: RAGChatRequest):
    """
    Chat with uploaded documents using RAG.

    Searches the vector database for relevant context and responds using LLM.
    """
    try:
        # Search for relevant context in Qdrant
        search_results = await QdrantService.search(
            query=request.message,
            limit=request.top_k,
            file_id=request.file_id,
        )

        # Format context from search results
        context = ""
        sources = []
        if search_results:
            for i, result in enumerate(search_results, 1):
                context += f"[{i}] {result['text']}\n\n"
                sources.append(
                    {
                        "text": result["text"],
                        "score": result["score"],
                        "metadata": result.get("metadata", {}),
                    }
                )

        if not context.strip():
            context = "No relevant information found in the uploaded documents."

        # Generate response using OpenAI
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": RAG_SYSTEM_PROMPT_TEMPLATE.format(context=context),
                },
                {"role": "user", "content": request.message},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        return RAGChatResponse(
            response=response.choices[0].message.content,
            sources=sources if search_results else None,
        )

    except Exception as e:
        print(f"RAG chat error: {e}")
        return RAGChatResponse(
            response="I'm having trouble accessing the document information right now. Please try again.",
            sources=None,
        )


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete all documents associated with a file ID."""
    try:
        await QdrantService.delete_by_file_id(file_id)
        return {"success": True, "message": f"Deleted all documents for file {file_id}"}
    except Exception as e:
        print(f"Delete error: {e}")
        return {"success": False, "message": f"Error deleting file: {str(e)}"}


@router.delete("/clear")
async def clear_all():
    """Clear all documents from the vector database."""
    try:
        await QdrantService.clear_all()
        return {"success": True, "message": "Cleared all documents from vector database"}
    except Exception as e:
        print(f"Clear error: {e}")
        return {"success": False, "message": f"Error clearing database: {str(e)}"}
