from fastapi import FastAPI
from contextlib import asynccontextmanager
import app.services.rag as rag 

@asynccontextmanager
async def lifespan(app: FastAPI):

    await rag.load_docs()
    yield
 

app = FastAPI(lifespan=lifespan)


from app.routers import ingest, rag_api,extract
app.include_router(ingest.router, prefix="/api", tags=["Ingest"])
app.include_router(extract.router, prefix="/api", tags=["Extract"])
app.include_router(rag_api.router, prefix="/api", tags=["RAG"])


@app.get("/healthz")
def health_check():
    return {"status": "ok"}




