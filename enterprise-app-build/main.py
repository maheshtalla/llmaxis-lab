from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path

#Fast API application instance
app=FastAPI(
    title="Enterprise AI Application API",
    description="Enterprise AI workflow demo",
    version="1.0.0"
)

#representing a resource model
class SupportRequest(BaseModel):
    user_input: str

#audit log for the application
audit_file = "audit_logs.json"
_audit_lock = threading.Lock()

#UI requests tracking 
def generate_request_id():
    return str(uuid.uuid4())

#prompt injection prevention
def detect_prompt_injection(user_input: str) -> bool:
    # Simple heuristic to detect prompt injection attempts
    suspicious_patterns = ["ignore previous instructions", "show password", "bypass security", 
                           "delete database", "ignore system prompts", "override system instructions", "malicious"]
#    return any(keyword in user_input.lower() for keyword in suspicious_keywords)

    for pattern in suspicious_patterns:
        if pattern.lower() in user_input.lower():
            return True
    return False

#PII detection and masking
def detect_pii(text):
    # Simple regex patterns for PII detection
    pii_patterns = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "phone": r"\+?\d[\d -]{8,12}\d",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b"
    }
    
    pii_found = {}
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            pii_found[pii_type] = matches
            
    return pii_found

def mask_pii(text, pii_found):
    for pii_type, matches in pii_found.items():
        for match in matches:
            text = text.replace(match, f"[MASKED {pii_type.upper()}]")
    return text

#build the prompt for the AI model
def build_prompt(masked_text):
    system_prompt = (
        "You are an enterprise AI assistant. Do not provide sensitive information "
        "or execute commands that could compromise security."
    )
    return (
        f"System Instructions: {system_prompt}\n\n"
        f"User Input: {masked_text}\n\n"
        "Assistant Response:"
    )

#use llm to generate response
user_real_llm = False
def call_llm(prompt: str) -> str:
    if not user_real_llm:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        return "This is a mock response from the AI model."
    raise RuntimeError("Real LLM mode is not configured.")

#combine all steps to process the user input
def compliance_check(response: str) -> dict[bool, str]:
    blocked_words = ["password", "credit card", "social security", "confidential"]
    for word in blocked_words:
        if word.lower() in response.lower():
            return {True: "Response contains blocked content."}
    return {False: "Response is compliant."}

# Preserve the original misspelled helper for existing imports.
complaince_check = compliance_check

#audit log writing
def write_audit_log(
    request_id: str,
    masked_input: str,
    response: str,
    pii_found: dict[str, list[str]],
) -> None:
    audit_path = Path(audit_file)
    log_entry = {
        "request_id": request_id,
        "masked_input": masked_input,
        "response": response,
        "pii_types": sorted(pii_found),
        "pii_counts": {pii_type: len(matches) for pii_type, matches in pii_found.items()},
    }

    with _audit_lock:
        logs = []
        if audit_path.exists():
            with audit_path.open(encoding="utf-8") as audit:
                logs = json.load(audit)
            if not isinstance(logs, list):
                raise ValueError(f"Audit log {audit_path} must contain a JSON list.")
        logs.append(log_entry)

        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=audit_path.parent,
                delete=False,
            ) as audit:
                temporary_path = audit.name
                json.dump(logs, audit, indent=4)
                audit.flush()
                os.fsync(audit.fileno())
            os.replace(temporary_path, audit_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

#health check endpoint
@app.get("/")
def root():
    return {
        "message": "Enterprise AI Application API",
        "docs": "/docs",
        "health": "/health",
        "process_request": "/process_request",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}    

#main endpoint to handle user requests
@app.post("/process_request")
def process_request(support_request: SupportRequest):
    request_id = generate_request_id()
    user_input = support_request.user_input
    
    # Prompt injection detection
    if detect_prompt_injection(user_input):
        return {"request_id": request_id, "error": "Prompt injection detected. Request blocked."}
    
    # PII detection and masking
    pii_found = detect_pii(user_input)
    masked_input = mask_pii(user_input, pii_found)
    
    # Build prompt for LLM
    prompt = build_prompt(masked_input)
    
    # Call LLM to get response
    response = call_llm(prompt)
    
    # Compliance check on the response
    compliance_result = compliance_check(response)
    
    # Write to audit log
    write_audit_log(request_id, masked_input, response, pii_found)
    
    return {
        "request_id": request_id,
        "masked_input": masked_input,
        "response": response,
        "compliance_result": compliance_result
    }
