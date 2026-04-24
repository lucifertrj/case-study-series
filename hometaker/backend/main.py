from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_TITLE, API_VERSION, CORS_ORIGINS
from backend.routes import chat_router, logs_router, preferences_router, rag_router

app = FastAPI(title=API_TITLE, version=API_VERSION)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(preferences_router)
app.include_router(logs_router)
app.include_router(chat_router)
app.include_router(rag_router)


@app.get("/")
async def root():
    return {"message": "HOME-TAKER API is running", "version": API_VERSION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
