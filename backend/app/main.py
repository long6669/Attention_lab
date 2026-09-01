import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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


frontend_directory = os.getenv("ATTNLAB_FRONTEND_DIR")
if frontend_directory:
    frontend_path = Path(frontend_directory)
    if not frontend_path.is_dir():
        raise RuntimeError(f"ATTNLAB_FRONTEND_DIR does not exist: {frontend_path}")
    app.mount(
        "/",
        StaticFiles(directory=frontend_path, html=True),
        name="frontend",
    )
