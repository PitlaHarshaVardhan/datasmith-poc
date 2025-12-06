import json
from pathlib import Path
from typing import List, Optional
from .models import PatientReport
import logging

DATA_DIR = Path(__file__).parent / "data"
PATIENTS_FILE = DATA_DIR / "patients.json"

logger = logging.getLogger(__name__)

def load_patients() -> List[PatientReport]:
    with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    patients = [PatientReport(**p) for p in raw]
    logger.info("Loaded %d patient records", len(patients))
    return patients

_PATIENT_CACHE: Optional[List[PatientReport]] = None

def get_patient_reports() -> List[PatientReport]:
    global _PATIENT_CACHE
    if _PATIENT_CACHE is None:
        _PATIENT_CACHE = load_patients()
    return _PATIENT_CACHE

def find_patient_by_name(name: str) -> Optional[PatientReport]:
    """
    Case-insensitive lookup by full name.
    Handles multiple hits with logging.
    """
    logger.info("Patient lookup attempt for name='%s'", name)
    patients = get_patient_reports()
    matches = [p for p in patients if p.patient_name.lower() == name.lower()]

    if not matches:
        logger.warning("No patient found for name='%s'", name)
        return None
    if len(matches) > 1:
        logger.warning("Multiple patients found for name='%s'", name)
        # In real system, disambiguate, but for POC return first.
    report = matches[0]
    logger.info("Patient found: %s", report)
    return report
