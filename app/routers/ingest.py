from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import uuid, os, json
from app.services.pdf_loader import extract_text_from_pdf

router = APIRouter()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@router.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """
    Upload 1..n PDF files, extract text, and save locally.
    Returns the document_id for each PDF for use in /api/extract.
    """
    documents = []

    for file in files:
        try:
            document_id = uuid.uuid4().hex
            filename = f"{file.filename}"
            file_path = os.path.join(DATA_DIR, filename)

            # Save the uploaded PDF
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Extract text content
            extracted_text = extract_text_from_pdf(file_path)

            # Save extracted text to JSON
            json_path = os.path.join(DATA_DIR, f"{document_id}.json")
            with open(json_path, "w", encoding="utf-8") as j:
                json.dump({"document_id": document_id, "content": extracted_text}, j, ensure_ascii=False, indent=2)

            documents.append({
                "document_id": document_id,
                "filename": file.filename,
                "stored_path": file_path,
                "json_path": json_path
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to ingest {file.filename}: {e}")

    return {"status": "success", "documents": documents}


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_form():
    """HTML form to upload PDFs"""
    return """
    <html>
        <head>
            <title>Upload PDF for Ingestion</title>
        </head>
        <body>
            <h2>Upload PDF files for ingestion</h2>
            <form action="/api/ingest" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
