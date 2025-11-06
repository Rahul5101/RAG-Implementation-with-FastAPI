





import os
import asyncio
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.db.sqlite.sqlite import SqliteDb
from agno.knowledge.reader.pdf_reader import PDFReader

# Load environment variables
load_dotenv()

# Define paths
PDF_DIR = "./data"  # Folder containing your PDFs
VECTOR_DB_PATH = "./local_lancedb"
SQLITE_DB_PATH = "./local_knowledge.sqlite"

# Global references
knowledge = None
model = None
agent = None


async def load_docs():
    """Initialize RAG pipeline and embed only PDFs from ./app/data."""
    global knowledge, model, agent

    if not os.path.exists(PDF_DIR):
        print(f"⚠️ Folder not found: {PDF_DIR}. Please upload your PDFs first.")
        return

    print(" Loading and indexing PDFs from:", PDF_DIR)

    # --- Create Embedder ---
    embedder = GeminiEmbedder(
        id="models/text-embedding-004",
        api_key=os.getenv("GOOGLE_API_KEY"),
        dimensions=768
    )

    # --- Create or Reinitialize Hybrid Vector DB ---
    vector_db = LanceDb(
        table_name="contracts_rag_store",
        uri=VECTOR_DB_PATH,
        search_type=SearchType.hybrid,
        embedder=embedder
    )

    # --- Create Knowledge Base (SQL Metadata + Vectors) ---
    knowledge = Knowledge(
        name="Contracts Knowledge Base",
        description="Knowledge base built solely from PDFs under app/data.",
        vector_db=vector_db,
        contents_db=SqliteDb(db_file=SQLITE_DB_PATH)
    )

    reader = PDFReader(chunk=True, chunk_size=1000, overlap=200)


    await knowledge.add_content_async(
        name="Contracts",
        path=PDF_DIR,
        include=["*.pdf"],  # ensures only PDFs are loaded
        reader=reader,
        metadata={"source": "contracts_folder"}
    )

    model = Gemini(
        id="gemini-2.0-flash",  # supported for content generation
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
        max_output_tokens=2048
    )

    agent = Agent(
        model=model,
        knowledge=knowledge,
        markdown=True,
        description="A contract-focused assistant limited to app/data PDFs.",
        instructions=[
            "Use only given pdf information available from the PDF knowledge base.",
            "Be precise, concise, and factual.",
            "If an answer cannot be found, say so explicitly.",
            "Always refer to the given pdf section or document if available.",
        ],
    )

    print(" Knowledge base and model initialized successfully!")


async def get_answer(query: str) -> str:
    """Query the RAG system for an answer based only on loaded PDFs."""
    global agent

    if not agent:
        raise RuntimeError("RAG not initialized. Call load_docs() first.")

    response_text = ""
    try:
        stream = agent.run(query, stream=True)
        for chunk in stream:
            if hasattr(chunk, "content") and chunk.content:
                response_text += chunk.content
    except Exception as e:
        response_text = f" Error during response generation: {str(e)}"

    return response_text.strip()
