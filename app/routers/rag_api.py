


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Iterator
import app.services.rag as rag  # Import your updated rag.py module
from agno.agent import RunEvent,RunOutputEvent

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import asyncio
from app.services.rag import get_answer 
import re

router = APIRouter()

# Model for the user query
class QueryRequest(BaseModel):
    question: str

@router.post("/rag")
async def ask_rag(request: QueryRequest):
    """Ask a natural language question using the RAG pipeline."""

    # Check if RAG system is initialized
    if not rag.agent:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please run /api/ingest first or ensure PDFs are loaded."
        )

    try:
        response_text = ""
        # Stream the answer from the agent
        stream: Iterator[RunOutputEvent] = rag.agent.run(request.question, stream=True)
        for chunk in stream:
            if chunk.event == RunEvent.run_content and getattr(chunk, "content", None):
                response_text += chunk.content

        if not response_text:
            response_text = "❌ No answer found in the contract knowledge base."

        return {
            "question": request.question,
            "answer": response_text.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating answer: {str(e)}")


@router.get("/rag", response_class=HTMLResponse)
async def get_rag_form():
    return """
    <html>
        <head>
            <title>RAG Query</title>
        </head>
        <body>
            <h2>Ask a question:</h2>
            <input type="text" id="question" placeholder="Enter your question" size="50">
            <button onclick="submitQuestion()">Ask</button>
            <h3>Answer:</h3>
            <div id="answer"></div>

            <script>
                async function submitQuestion() {
                    const question = document.getElementById("question").value;
                    const responseDiv = document.getElementById("answer");

                    const response = await fetch("/api/rag", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({ question: question })
                    });

                    if(response.ok) {
                        const data = await response.json();
                        responseDiv.innerHTML = "<b>" + data.answer + "</b>";
                    } else {
                        const error = await response.json();
                        responseDiv.innerHTML = "<b style='color:red'>Error: " + JSON.stringify(error) + "</b>";
                    }
                }
            </script>
        </body>
    </html>
    """

# POST: Accept JSON and return answer
@router.post("/rag")
async def rag_query(query: QueryRequest):
    try:
        answer = await get_answer(query.question)

        # Remove references like (Contracts, page XX)
        clean_answer = re.sub(r"\(Contracts, page \d+\)", "", answer)
        # Remove extra spaces caused by removals
        clean_answer = re.sub(r"\s{2,}", " ", clean_answer).strip()

        return {"question": query.question, "answer": clean_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))