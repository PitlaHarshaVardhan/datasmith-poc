from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

from .logging_config import setup_logging
from .models import ChatRequest, ChatResponse
from .agents import ReceptionistAgent, ClinicalAgent, get_session

from .db import get_patient_reports



setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Post-Discharge Medical AI Assistant (POC)",
    description="DataSmith AI – GenAI Intern Assignment POC",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for POC; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/patients")
def list_patients():
    return {"patients": [p.patient_name for p in get_patient_reports()]}


@app.get("/")
def root():
    # Small redirect hint for running
    return {"message": "Backend running. Open /static/index.html in browser."}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info("Incoming message. session_id=%s, message='%s'", req.session_id, req.message)

    state = get_session(req.session_id)
    stage = state["stage"]

    # Decide which agent should handle the message
    if stage in ("ask_name", "await_name", "have_patient"):
        first_response = ReceptionistAgent.handle_message(req.session_id, req.message)
        # If handoff to clinical triggered
        if first_response["metadata"].get("handoff"):
            # Immediately ask clinical agent to handle the same message again, or wait for next user message.
            clinical_response = ClinicalAgent.handle_message(req.session_id, req.message)
            # Return the clinical answer as the final response
            logger.info("Returning clinical response after handoff.")
            return ChatResponse(
                session_id=req.session_id,
                reply=clinical_response["reply"],
                agent=clinical_response["agent"],
                metadata=clinical_response["metadata"],
            )
        else:
            return ChatResponse(
                session_id=req.session_id,
                reply=first_response["reply"],
                agent=first_response["agent"],
                metadata=first_response["metadata"],
            )

    # Already in clinical stage
    if stage == "clinical":
        clinical_response = ClinicalAgent.handle_message(req.session_id, req.message)
        return ChatResponse(
            session_id=req.session_id,
            reply=clinical_response["reply"],
            agent=clinical_response["agent"],
            metadata=clinical_response["metadata"],
        )

    # Fallback
    reply = "Something went wrong in routing your message. Please start a new session."
    logger.error("Unknown stage '%s' for session_id=%s", stage, req.session_id)
    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        agent="system",
        metadata={"stage": stage},
    )
