from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    agent: str  # "receptionist" or "clinical"
    metadata: Dict[str, Any] = {}

class PatientReport(BaseModel):
    patient_name: str
    discharge_date: str
    primary_diagnosis: str
    medications: List[str]
    dietary_restrictions: str
    follow_up: str
    warning_signs: str
    discharge_instructions: str

class RetrievedDoc(BaseModel):
    content: str
    source: str
    score: float

class RAGResult(BaseModel):
    answer: str
    docs: List[RetrievedDoc]
    used_web_search: bool
    web_sources: Optional[List[str]] = None
