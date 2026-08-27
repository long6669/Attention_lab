from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as attention_router

app = FastAPI(
    title="AttnLab API",
    version="0.1.0",
    description="Executable attention graphs for the AttnLab MVP.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(attention_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
