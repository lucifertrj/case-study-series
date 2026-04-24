# HOME-TAKER 🏠

> **AI-powered "External Brain" for caregivers with RAG-powered document chat**

There are over **53 million unpaid caregivers** in the US providing **$600 billion** in labor annually, with **61%** of them working full-time while managing complex health and home logistics. Globally, **1 in 5 adults** are informal caregivers, spending **20+ hours/week** managing medications, appointments, and care coordination for aging family members.

**HOME-TAKER** is an AI-powered "External Brain" that uses **Qdrant vector search** to proactively track and connect micro-details like medication changes, trusted service providers, and family milestones. Upload documents (PDFs, text files) and chat with them using RAG (Retrieval-Augmented Generation).

---

## Project Structure

```
home-taker/
├── backend/          # FastAPI backend with Qdrant RAG
│   ├── routes/       # API endpoints
│   ├── services/     # Business logic (Qdrant, Document Processing)
│   ├── models.py     # Data models
│   ├── config.py     # Configuration
│   └── main.py       # Application entry point
├── frontend/         # React + Vite frontend
│   ├── src/          # React components
│   └── public/       # Static assets
├── pyproject.toml    # Python dependencies
└── README.md
```

---

## Setup Guide

### Prerequisites

- **Python 3.12+**
- **Node.js 18+** (for frontend)
- **uv** package manager (recommended) or pip

### 1. Clone the Repository

```bash
git clone <repository-url>
cd home-taker
```

### 2. Backend Setup

#### Install Python dependencies

Using **uv** (recommended):
```bash
uv sync
```

Or using **pip**:
```bash
pip install -e .
```

#### Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key
LLM_API_KEY=your_openai_api_key
OPENAI_API_KEY=your_openai_api_key

# Qdrant Cloud Configuration
QDRANT_HOST=your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# Embedding Configuration
EMBEDDING_MODEL=jinaai/jina-embeddings-v3
EMBEDDING_DIMENSION=1024
```

#### Run the Backend

```bash
# From project root
uv run python -m backend.main

# Or with pip
python -m backend.main
```

The API will be available at `http://localhost:8000`

**Test the API:**
```bash
curl http://localhost:8000
```

### 4. Frontend Setup

#### Navigate to frontend directory

```bash
cd frontend
```

#### Install dependencies

```bash
npm install
```

#### Configure Environment (if needed)

Check `frontend/.env.example` (if available) or create `.env`:

```bash
VITE_API_URL=http://localhost:8000
```

#### Run the Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

---

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/preferences` | GET/POST | Manage user preferences |
| `/api/logs` | GET/POST | Access and create logs |
| `/api/chat` | POST | Chat with AI assistant |

### RAG Endpoints (Document Chat)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rag/upload` | POST | Upload file (PDF/TXT/MD) for RAG |
| `/api/rag/chat` | POST | Chat with uploaded documents |
| `/api/rag/files/{file_id}` | DELETE | Delete documents for a file |
| `/api/rag/clear` | DELETE | Clear all documents |

### Example: Upload a Document

```bash
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "success": true,
  "message": "Successfully processed 15 chunks from document.pdf",
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_count": 15
}
```

### Example: Chat with Documents

```bash
curl -X POST http://localhost:8000/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What medications are mentioned in the document?",
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "top_k": 5
  }'
```

Response:
```json
{
  "response": "The document mentions the following medications...",
  "sources": [
    {
      "text": "Medication: Lisinopril 10mg...",
      "score": 0.89,
      "metadata": {"file_id": "..."}
    }
  ]
}
```

---

## Development

### Backend

```bash
# Run with auto-reload
uv run python -m backend.main

# Run tests (if available)
uv run pytest
```

### Frontend

```bash
# Development server with hot-reload
npm run dev

# Build for production
npm run build

# Lint code
npm run lint
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 💙 Support

HOME-TAKER is built to support caregivers everywhere. If you're using this project and need help, please open an issue.

**Remember: You're not alone in this caregiving journey.**
