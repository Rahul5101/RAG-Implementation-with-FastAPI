import requests
import os

WEBHOOK = os.environ.get('WEBHOOK_URL')

def emit_event(event_type: str, payload: dict):
    if not WEBHOOK:
        return
    try:
        requests.post(WEBHOOK, json={'type': event_type, 'payload': payload}, timeout=5)
    except Exception:
        pass