# import os
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from pathlib import Path

# # Load model
# MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# # Define data directory
# DATA_DIR = Path('/app/data') if Path('/app/data').exists() else Path('./data')

# class SimpleRAG:
#     def __init__(self):
#         self.index = None
#         self.passages = []
#         self.embeddings = None

#     def build_index_from_docs(self):
#         docs_dir = DATA_DIR / 'docs'
#         if not docs_dir.exists():
#             print("Docs directory not found.")
#             return

#         self.passages = []
#         for p in docs_dir.glob('*.txt'):
#             text = p.read_text(encoding='utf-8')
#             chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#             for i, c in enumerate(chunks):
#                 self.passages.append({
#                     'doc': p.name.replace('.txt', ''),
#                     'chunk_idx': i,
#                     'text': c
#                 })

#         if not self.passages:
#             print("No text chunks found.")
#             return

#         texts = [p['text'] for p in self.passages]
#         self.embeddings = MODEL.encode(texts, convert_to_numpy=True)

#         d = self.embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(d)
#         self.index.add(self.embeddings)
#         print(f"Indexed {len(self.passages)} text chunks.")

#     def query(self, question: str, top_k=3):
#         if self.index is None:
#             print("Index has not been built yet.")
#             return []

#         q_emb = MODEL.encode([question], convert_to_numpy=True)
#         D, I = self.index.search(q_emb, top_k)
#         results = [self.passages[idx] for idx in I[0]]
#         return results

# RAG = SimpleRAG()

# # Build index at startup if docs exist
# try:
#     RAG.build_index_from_docs()
# except Exception as e:
#     print("Error building index:", e)

# # Now query the index
# answers = RAG.query("What is SQL?", top_k=3)
# print(answers)



# import os
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from pathlib import Path
# import fitz  # PyMuPDF

# MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# DATA_DIR = Path('/app/data') if Path('/app/data').exists() else Path('./data')

# class SimpleRAG:
#     def __init__(self):
#         self.index = None
#         self.passages = []
#         self.embeddings = None

#     def build_index_from_pdfs(self):
#         docs_dir = DATA_DIR
#         if not docs_dir.exists():
#             print("Data directory not found.")
#             return

#         self.passages = []
#         pdf_files = list(docs_dir.glob('*.pdf'))
#         if not pdf_files:
#             print("No PDF files found in data directory.")
#             return

#         for pdf_file in pdf_files:
#             try:
#                 doc = fitz.open(pdf_file)
#                 full_text = ""
#                 for page in doc:
#                     full_text += page.get_text("text")
#                 doc.close()

#                 # Break text into chunks of roughly 1000 characters
#                 chunks = [full_text[i:i + 1000] for i in range(0, len(full_text), 1000)]
#                 for i, chunk in enumerate(chunks):
#                     self.passages.append({
#                         'doc': pdf_file.name.replace('.pdf', ''),
#                         'chunk_idx': i,
#                         'text': chunk
#                     })
#             except Exception as e:
#                 print(f"Error reading {pdf_file.name}: {e}")

#         if not self.passages:
#             print("No text extracted from PDFs.")
#             return

#         texts = [p['text'] for p in self.passages]
#         self.embeddings = MODEL.encode(texts, convert_to_numpy=True)

#         d = self.embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(d)
#         self.index.add(self.embeddings)
#         print(f"Indexed {len(self.passages)} PDF text chunks from {len(pdf_files)} files.")

#     def query(self, question: str, top_k=3):
#         if self.index is None:
#             print("Index has not been built yet.")
#             return []
#         q_emb = MODEL.encode([question], convert_to_numpy=True)
#         D, I = self.index.search(q_emb, top_k)
#         results = [self.passages[idx] for idx in I[0]]
#         return results

# # Instantiate and index
# RAG = SimpleRAG()
# RAG.build_index_from_pdfs()

# # Example query
# answers = RAG.query("What is databases?", top_k=1)
# for i, ans in enumerate(answers, 1):
#     # print(f"[{i}] Document: {ans['doc']}, Chunk: {ans['chunk_idx']}")
#     print(ans['text'], "\n---")

"""
RAG-based Chatbot using Agno v2.2.1
Fully local & modular — processes PDFs from ./data folder
"""

"""
RAG Chatbot using Agno v2.2.1 and Google Gemini.
Reads PDFs from ./data and answers questions with file‑aware retrieval.
"""

"""
RAG Chatbot (Agno v2.2.1)
Now uses PDFKnowledgeBase from agno.knowledge.reader.pdf_reader
"""
# import os, asyncio
# from agno.agent import Agent
# from agno.models.google import Gemini
# from agno.knowledge.knowledge import Knowledge
# from agno.vectordb.lancedb import LanceDb, SearchType
# from agno.knowledge.embedder.google import GeminiEmbedder
# from agno.db.sqlite.sqlite import SqliteDb
# from dotenv import load_dotenv
# load_dotenv()

# # config
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your_google_api_key")

# # 1. Set up embedding model and vector store
# embedder = GeminiEmbedder(
#     id="models/text-embedding-004",
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     dimensions=768
# )
# vector_db = LanceDb(
#     table_name="contracts_rag_store",
#     uri="./local_lancedb",
#     search_type=SearchType.hybrid,
#     embedder=embedder
# )
# contents_db = SqliteDb(db_file="./local_knowledge.sqlite")


# # 2. Knowledge object
# knowledge = Knowledge(
#     name="Contracts RAG Knowledge",
#     description="Knowledge base for contract documents",
#     vector_db=vector_db,
#     contents_db=contents_db
# )

# # # 3. Load PDF contents and index them
# import asyncio
# from agno.knowledge.knowledge import Knowledge
# from agno.knowledge.reader.pdf_reader import PDFReader

# async def load_docs():
#     reader = PDFReader(chunk=True, chunk_size=1000, overlap=200)
#     await knowledge.add_content_async(
#         name="Contracts",
#         path="./data",
#         reader=reader,
#         metadata={"source": "contracts_folder"}
#     )
#     print("✅ Documents loaded with chunking.")

# asyncio.run(load_docs())

# # 4. Use Gemini model for responses
# model = Gemini(
#     id="gemini-2.0-flash",
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.3,
#     max_output_tokens=2048
# )

# # 5. Create Agent for RAG
# agent = Agent(
#     model=model,
#     knowledge=knowledge,
#     markdown=True,
#     description="An assistant that answers using loaded contract documents.",
#     instructions=[
#         "Base answers on context retrieved from the knowledge base.",
#         "Be precise and clear.",
#         "If answer cannot be found, admit it.",
#         "Cite sources and sections when applicable."
#     ]
# )
# from typing import Iterator
# from agno.agent import Agent, RunOutput, RunOutputEvent, RunEvent
# # response = agent.run("tell me what is deadlocks.")
# # print(response.content)

# stream: Iterator[RunOutputEvent] = agent.run("what is round robin process", stream=True)
# for chunk in stream:
#     if chunk.event == RunEvent.run_content:
#         print(chunk.content)






# import os
# import asyncio
# from dotenv import load_dotenv
# from agno.agent import Agent
# from agno.models.google import Gemini
# from agno.knowledge.knowledge import Knowledge
# from agno.vectordb.lancedb import LanceDb, SearchType
# from agno.knowledge.embedder.google import GeminiEmbedder
# from agno.db.sqlite.sqlite import SqliteDb
# from agno.knowledge.reader.pdf_reader import PDFReader

# load_dotenv()  # Load environment variables from .env

# # Paths
# PDF_DIR = "./app/data"
# VECTOR_DB_PATH = "./local_lancedb"
# SQLITE_DB_PATH = "./local_knowledge.sqlite"

# # Global instances (initialized during startup)
# knowledge = None
# model = None
# agent = None


# async def load_docs():
#     """Loads contract PDFs and initializes the RAG components."""
#     global knowledge, model, agent

#     # --- Initialize Embedder and Vector DB ---
#     embedder = GeminiEmbedder(
#         id="models/text-embedding-004",
#         api_key=os.getenv("GOOGLE_API_KEY"),
#         dimensions=768
#     )

#     vector_db = LanceDb(
#         table_name="contracts_rag_store",
#         uri=VECTOR_DB_PATH,
#         search_type=SearchType.hybrid,
#         embedder=embedder
#     )

#     # --- Initialize Knowledge Base ---
#     knowledge = Knowledge(
#         name="Contracts Knowledge Base",
#         description="Knowledge base for contract documents.",
#         vector_db=vector_db,
#         contents_db=SqliteDb(db_file=SQLITE_DB_PATH)
#     )

#     # --- Read PDFs ---
#     if not os.path.exists(PDF_DIR):
#         print(f"⚠️ Folder not found: {PDF_DIR}. Please upload contract PDFs first.")
#         return

#     reader = PDFReader(chunk=True, chunk_size=1000, overlap=200)
#     await knowledge.add_content_async(
#         name="Contracts",
#         path=PDF_DIR,
#         reader=reader,
#         metadata={"source": "contracts_folder"}
#     )

#     # --- Initialize Gemini Model ---
#     model = Gemini(
#         id="gemini-2.0-flash",
#         api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.3,
#         max_output_tokens=2048
#     )

#     # --- Create Agent ---
#     agent = Agent(
#         model=model,
#         knowledge=knowledge,
#         markdown=True,
#         description="An assistant that answers based on contract documents.",
#         instructions=[
#             "Answer based on the retrieved contract context.",
#             "Be precise, concise, and factual.",
#             "If you cannot find the answer, clearly state that.",
#             "Always refer to the relevant section or clause if possible."
#         ],
#     )

#     print("✅ Knowledge base and model loaded successfully!")


# async def get_answer(query: str) -> str:
#     """Run a query through the RAG agent and return the answer."""
#     global agent

#     if not agent:
#         raise RuntimeError(
#             "RAG system not initialized yet. Please reload the app or run /api/ingest first."
#         )

#     response_text = ""
#     try:
#         stream = agent.run(query, stream=True)
#         for chunk in stream:
#             if hasattr(chunk, "content"):
#                 response_text += chunk.content
#     except Exception as e:
#         response_text = f"Error while generating answer: {str(e)}"

#     return response_text.strip()





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
