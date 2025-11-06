# extractor.py

import pandas as pd
import re
from typing import Dict, Any

class ContractExtractor:
    """Extractor for structured fields from contract text using regex and heuristics."""

    
def __init__(self):
    pass

def extract_fields(self, text: str) -> Dict[str, Any]:
    """Extracts structured fields from the contract text using regex and pandas where applicable."""

    data = {
        "parties": self._extract_parties(text),
        "effective_date": self._extract_effective_date(text),
        "term": self._extract_term(text),
        "governing_law": self._extract_governing_law(text),
        "payment_terms": self._extract_payment_terms(text),
        "termination": self._extract_termination(text),
        "auto_renewal": self._extract_auto_renewal(text),
        "confidentiality": self._extract_confidentiality(text),
        "indemnity": self._extract_indemnity(text),
        "liability_cap": self._extract_liability_cap(text),
        "signatories": self._extract_signatories(text),
    }

    # Using pandas DataFrame for structured output (optional but useful)
    df = pd.DataFrame([data])
    return df.to_dict(orient="records")[0]

# ---------------- Extraction Helpers ----------------

def _extract_parties(self, text: str):
    match = re.findall(r"between\s+(.*?)\s+and\s+(.*?)\.", text, re.IGNORECASE)
    if match:
        return list(match[0])
    return []

def _extract_effective_date(self, text: str):
    match = re.search(r"effective as of\s+(\w+\s+\d{1,2},\s+\d{4})", text, re.IGNORECASE)
    return match.group(1) if match else None

def _extract_term(self, text: str):
    match = re.search(r"term of this agreement shall be\s+(.*?)\.", text, re.IGNORECASE)
    return match.group(1) if match else None

def _extract_governing_law(self, text: str):
    match = re.search(r"governed by the laws of\s+(.*?)\.", text, re.IGNORECASE)
    return match.group(1) if match else None

def _extract_payment_terms(self, text: str):
    match = re.search(r"payment terms?.*?(\d+\s*days)", text, re.IGNORECASE)
    return match.group(1) if match else None

def _extract_termination(self, text: str):
    match = re.search(r"termination.*?may terminate.*?", text, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip() if match else None

def _extract_auto_renewal(self, text: str):
    if re.search(r"auto[-\s]?renewal", text, re.IGNORECASE):
        return True
    return False

def _extract_confidentiality(self, text: str):
    match = re.search(r"confidentiality.*?obligations.*?", text, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip() if match else None

def _extract_indemnity(self, text: str):
    match = re.search(r"indemnif(y|ication).*?", text, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip() if match else None

def _extract_liability_cap(self, text: str):
    match = re.search(r"liability shall not exceed\s+([\$\d,]+)\s*(USD|INR|EUR|GBP)?", text, re.IGNORECASE)
    if match:
        amount = match.group(1).replace(",", "")
        currency = match.group(2) or "USD"
        return {"amount": float(amount.replace("$", "")), "currency": currency}
    return {"amount": None, "currency": None}

def _extract_signatories(self, text: str):
    matches = re.findall(r"Signed by:\s*(.*?)\s*,\s*(.*?)\n", text, re.IGNORECASE)
    return [{"name": m[0], "title": m[1]} for m in matches]

