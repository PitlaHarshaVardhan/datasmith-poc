import logging
from typing import Dict, Any, List
import os

import google.generativeai as genai

from .db import find_patient_by_name, get_patient_reports
from .rag import generate_answer_with_rag
from .web_search import web_search
from .models import PatientReport, RAGResult

logger = logging.getLogger(__name__)

# ----- Configure Gemini -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set – agents will fail until it is configured.")

genai.configure(api_key=GEMINI_API_KEY)

CHAT_MODEL = "gemini-1.5-flash"  # or "gemini-pro"

SESSION_STATE: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSION_STATE:
        SESSION_STATE[session_id] = {
            "stage": "ask_name",   # "ask_name", "await_name", "have_patient", "clinical"
            "patient": None,
            "history": [],
        }
    return SESSION_STATE[session_id]


# ---------- Receptionist Agent ----------

class ReceptionistAgent:
    name = "receptionist"

    @staticmethod
    def handle_message(session_id: str, message: str) -> Dict[str, Any]:
        state = get_session(session_id)
        stage = state["stage"]
        logger.info("Receptionist handling message. stage=%s, message='%s'", stage, message)

        # Log history
        state["history"].append({"role": "patient", "message": message})

        # Stage 1: ask for name (INIT message)
        if stage == "ask_name":
            state["stage"] = "await_name"
            reply = "Hello! I'm your post-discharge care assistant. What's your full name?"
            return {"reply": reply, "agent": ReceptionistAgent.name, "metadata": {"stage": state["stage"]}}

        # Stage 2: receiving name
        if stage == "await_name":
            patient_name = message.strip()
            report = find_patient_by_name(patient_name)
            if report is None:
                # Suggest available patients so a new tester can use the app
                all_patients: List[PatientReport] = get_patient_reports()
                patient_names = [p.patient_name for p in all_patients]
                reply = (
                    f"I could not find a discharge report for '{patient_name}'.\n\n"
                    "Here are sample patients you can try:\n"
                    + "\n".join("- " + n for n in patient_names)
                )
                return {
                    "reply": reply,
                    "agent": ReceptionistAgent.name,
                    "metadata": {"stage": state["stage"], "patient_found": False},
                }

            # Found patient
            state["patient"] = report
            state["stage"] = "have_patient"
            reply = (
                f"Hi {report.patient_name}! I found your discharge report from {report.discharge_date} "
                f"for {report.primary_diagnosis}. How are you feeling today? "
                "Are you following your medication schedule?"
            )
            logger.info("Patient %s loaded into session %s", report.patient_name, session_id)
            return {
                "reply": reply,
                "agent": ReceptionistAgent.name,
                "metadata": {"stage": state["stage"], "patient": report.patient_name},
            }

        # Stage 3: have patient, classify question
        if stage == "have_patient":
            if ReceptionistAgent._is_medical_query(message):
                state["stage"] = "clinical"
                reply = (
                    "This sounds like a medical concern. "
                    "I'm connecting you to our Clinical AI Agent now."
                )
                logger.info("Routing to Clinical Agent for session %s", session_id)
                return {
                    "reply": reply,
                    "agent": ReceptionistAgent.name,
                    "metadata": {"handoff": True, "to": "clinical", "stage": state["stage"]},
                }
            else:
                reply = ReceptionistAgent._small_talk_response(state["patient"], message)
                return {
                    "reply": reply,
                    "agent": ReceptionistAgent.name,
                    "metadata": {"stage": state["stage"], "handoff": False},
                }

        # Stage 4: once in clinical, receptionist doesn't respond anymore
        if stage == "clinical":
            reply = (
                "You are now connected to the Clinical AI Agent. "
                "Please wait for a medical answer."
            )
            return {
                "reply": reply,
                "agent": ReceptionistAgent.name,
                "metadata": {"stage": state["stage"]},
            }

        # Fallback
        reply = "I'm not sure how to handle that. Could you rephrase?"
        return {"reply": reply, "agent": ReceptionistAgent.name, "metadata": {"stage": state["stage"]}}

    @staticmethod
    def _is_medical_query(message: str) -> bool:
        msg = message.lower()
        medical_keywords = [
            "pain", "swelling", "shortness of breath", "dizzy", "dizziness",
            "blood pressure", "bp", "urine", "kidney", "fever", "symptom",
            "should i be worried", "emergency", "side effect",
        ]
        return any(k in msg for k in medical_keywords)

    @staticmethod
    def _small_talk_response(report: PatientReport, message: str) -> str:
        # Simple templated answer (no LLM needed here)
        return (
            f"Thanks for the update, {report.patient_name}. "
            f"Remember your discharge instructions: {report.discharge_instructions}. "
            "If you have any medical symptoms like "
            f"{report.warning_signs}, please tell me and I'll connect you "
            "to the Clinical AI Agent.\n\n"
            "This is an AI assistant for educational purposes only.\n"
            "Always consult healthcare professionals for medical advice."
        )


# ---------- Clinical Agent ----------

class ClinicalAgent:
    name = "clinical"

    @staticmethod
    def handle_message(session_id: str, message: str) -> Dict[str, Any]:
        state = get_session(session_id)
        state["history"].append({"role": "patient", "message": message})
        patient: PatientReport = state.get("patient")

        logger.info(
            "Clinical agent handling message for session %s, patient=%s",
            session_id,
            getattr(patient, "patient_name", None),
        )

        # Build personalized context
        patient_context = ""
        if patient:
            patient_context = (
                f"Patient: {patient.patient_name}\n"
                f"Primary diagnosis: {patient.primary_diagnosis}\n"
                f"Medications: {', '.join(patient.medications)}\n"
                f"Dietary restrictions: {patient.dietary_restrictions}\n"
                f"Warning signs: {patient.warning_signs}\n\n"
            )

        question = patient_context + "Patient question: " + message

        use_web = ClinicalAgent._should_use_web_search(message)

        if use_web:
            web_results = web_search(message, max_results=3)
            rag_result: RAGResult = generate_answer_with_rag(question, k=4)
            answer = ClinicalAgent._answer_with_web_and_rag(question, rag_result, web_results)
            used_web = True
        else:
            rag_result: RAGResult = generate_answer_with_rag(question, k=4)
            web_results = []
            answer = ClinicalAgent._format_answer_from_rag(rag_result)
            used_web = False

        logger.info(
            "Clinical agent answer generated. used_web=%s, docs=%d, web_results=%d",
            used_web,
            len(rag_result.docs),
            len(web_results),
        )
        state["history"].append({"role": "clinical", "message": answer})

        metadata = {
            "used_web_search": used_web,
            "num_rag_docs": len(rag_result.docs),
            "web_sources": web_results[:3],
        }

        return {"reply": answer, "agent": ClinicalAgent.name, "metadata": metadata}

    @staticmethod
    def _should_use_web_search(message: str) -> bool:
        lower = message.lower()
        keywords = ["latest", "recent", "new research", "study", "guideline", "2024", "2025"]
        return any(k in lower for k in keywords)

    @staticmethod
    def _format_answer_from_rag(r: RAGResult) -> str:
        citations = []
        for i, d in enumerate(r.docs, start=1):
            citations.append(f"[Ref {i}] {d.source} (similarity: {d.score:.2f})")
        citation_text = "\n\nSources:\n" + "\n".join(citations)

        disclaimer = (
            "\n\nThis is an AI assistant for educational purposes only.\n"
            "Always consult healthcare professionals for medical advice."
        )
        return r.answer + citation_text + disclaimer

    @staticmethod
    def _answer_with_web_and_rag(question: str, rag_result: RAGResult, web_results: Any) -> str:
        # Build context
        context_rag = "\n\n".join(
            [f"[RAG {i+1}] {d.content}" for i, d in enumerate(rag_result.docs)]
        )
        context_web = "\n\n".join(
            [f"[WEB {i+1}] {w}" for i, w in enumerate(web_results)]
        )

        prompt = (
            "You are a Clinical AI assistant with access to:\n"
            "- Nephrology reference materials (RAG context)\n"
            "- Recent web search results (WEB context)\n\n"
            "You must:\n"
            "1. Clearly separate information from reference materials vs web search.\n"
            "2. Use phrases like 'According to reference materials [RAG]' and "
            "'According to recent web information [WEB]'.\n"
            "3. Provide a patient-friendly explanation.\n"
            "4. End with the standard medical disclaimers.\n\n"
            f"Patient question: {question}\n\n"
            f"RAG context:\n{context_rag}\n\n"
            f"Web search context:\n{context_web}\n\n"
            "Now provide a single, coherent answer."
        )

        model = genai.GenerativeModel(CHAT_MODEL)
        response = model.generate_content(prompt)
        base_answer = response.text or ""

        disclaimer = (
            "\n\nThis is an AI assistant for educational purposes only.\n"
            "Always consult healthcare professionals for medical advice."
        )
        return base_answer + disclaimer
