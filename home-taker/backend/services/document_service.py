import uuid
from pathlib import Path

from pypdf import PdfReader


class DocumentProcessingService:
    """Service for processing and chunking documents."""

    @staticmethod
    def extract_text_from_file(file_path: Path, file_type: str) -> str:
        """
        Extract text from a file based on its type.

        Args:
            file_path: Path to the file
            file_type: MIME type or file extension

        Returns:
            Extracted text content
        """
        if file_type.endswith("pdf") or file_path.suffix.lower() == ".pdf":
            return DocumentProcessingService._extract_from_pdf(file_path)
        elif file_type.endswith("txt") or file_path.suffix.lower() == ".txt":
            return DocumentProcessingService._extract_from_text(file_path)
        elif file_type.endswith("markdown") or file_path.suffix.lower() in [
            ".md",
            ".markdown",
        ]:
            return DocumentProcessingService._extract_from_text(file_path)
        else:
            # Default to text extraction
            return DocumentProcessingService._extract_from_text(file_path)

    @staticmethod
    def _extract_from_pdf(file_path: Path) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(file_path)
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_from_text(file_path: Path) -> str:
        """Extract text from plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> list[dict[str, str]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks

        Returns:
            List of dicts with 'id' and 'text' keys
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            # If we're not at the end, try to break at a sentence or word boundary
            if end < text_length:
                # Try to find a sentence boundary
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)

                if break_point > start:
                    end = break_point + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({"id": str(uuid.uuid4()), "text": chunk_text})

            start = end - chunk_overlap
            if start >= text_length:
                break

        # If no chunks were created but text exists, create one chunk
        if not chunks and text.strip():
            chunks.append({"id": str(uuid.uuid4()), "text": text.strip()})

        return chunks

    @staticmethod
    async def process_file(
        file_path: Path, file_type: str, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> list[dict[str, str]]:
        """
        Process a file: extract text and chunk it.

        Args:
            file_path: Path to the file
            file_type: MIME type or file extension
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks

        Returns:
            List of chunk dicts with 'id' and 'text' keys
        """
        text = DocumentProcessingService.extract_text_from_file(file_path, file_type)
        chunks = DocumentProcessingService.chunk_text(text, chunk_size, chunk_overlap)
        return chunks
