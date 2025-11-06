from fastapi import APIRouter, HTTPException,Request
from pydantic import BaseModel
import os, json
from fastapi.responses import HTMLResponse

router = APIRouter()

DATA_DIR = "data"

# Define a schema for input
class ExtractRequest(BaseModel):
    document_id: str

@router.post("/extract")
async def extract(request: ExtractRequest):
    document_id = request.document_id
    json_path = os.path.join(DATA_DIR, f"{document_id}.json")

    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Document not found")

    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Simulate extraction (replace with actual NLP extraction logic later)
    extracted_info = {
        "parties": [],
        "effective_date": None,
        "term": None,
        "governing_law": None,
        "payment_terms": None,
        "termination": None,
        "auto_renewal": None,
        "confidentiality": None,
        "indemnity": None,
        "liability_cap": None,
        "signatories": [],
        "content_summary": text[:1000]  # preview of extracted text
    }

    return {"document_id": document_id, "extracted_info": extracted_info}





@router.get("/extract", response_class=HTMLResponse)
async def extract_form(request: Request):
    return """
    <html>
        <head>
            <title>Extract Document Data</title>
        </head>
        <body>
            <h2>Extract Information from Uploaded PDF</h2>
            <form action="/api/extract" method="post" enctype="application/json" onsubmit="submitForm(event)">
                <label for="document_id">Enter Document ID:</label><br><br>
                <input type="text" id="document_id" name="document_id" required><br><br>
                <button type="submit">Extract</button>
            </form>
            <pre id="response"></pre>

            <script>
            async function submitForm(event) {
                event.preventDefault();
                const document_id = document.getElementById('document_id').value;
                const res = await fetch('/api/extract', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ document_id })
                });
                const data = await res.json();
                document.getElementById('response').textContent = JSON.stringify(data, null, 2);
            }
            </script>
        </body>
    </html>
    """