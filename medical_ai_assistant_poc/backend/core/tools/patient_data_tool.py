import json
import os
import logging
import re
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from langchain.tools import Tool
from difflib import SequenceMatcher

load_dotenv()

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "logs", "patient_retrieval.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/patient_data.db")
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
DATABASE_URL = f"sqlite:///{os.path.join(PROJECT_ROOT, DATABASE_PATH)}"

Base = declarative_base()

class PatientReport(Base):
    __tablename__ = "patient_reports"
    
    patient_id = Column(String, primary_key=True)
    name = Column(String, index=True, nullable=False)
    report_json = Column(Text, nullable=False)  # Changed from full_report_json to report_json

def get_db_engine():
    return create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )

class PatientDataTool:
    def __init__(self):
        self.engine = get_db_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        # Don't create tables here - they should already exist from setup_data.py

    def _get_session(self) -> Session:
        return self.SessionLocal()

    def _fuzzy_match_score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _format_result(self, status: str, data: Optional[Dict] = None, error: Optional[str] = None, suggestions: Optional[List[str]] = None, method: Optional[str] = None, similarity: Optional[float] = None) -> Dict[str, Any]:
        result = {
            "status": status,
            "data": data,
            "error": error,
            "suggestions": suggestions,
            "method": method,
            "similarity": similarity
        }
        return {k: v for k, v in result.items() if v is not None}

    def search_patients(self, query: str) -> Dict[str, Any]:
        logger.info(f"Attempting to retrieve patient with query: '{query}'")
        session = self._get_session()
        
        try:
            # Check if query is a Patient ID (e.g., P001)
            if re.match(r"P\d{3}", query.upper()):
                logger.info(f"Query '{query}' detected as Patient ID.")
                report = session.query(PatientReport).filter(
                    PatientReport.patient_id == query.upper()
                ).first()
                
                if report:
                    logger.info(f"Patient ID match found for {report.name}.")
                    return self._format_result(
                        status="success",
                        data=json.loads(report.report_json),  # Changed from full_report_json
                        method="exact_id_match"
                    )
                else:
                    logger.warning(f"Patient ID '{query}' not found.")
                    return self._format_result(
                        status="not_found",
                        error=f"Patient ID '{query}' not found in the database."
                    )

            # Try exact name match
            exact_match = session.query(PatientReport).filter(
                PatientReport.name.ilike(query)
            ).all()
            
            if len(exact_match) == 1:
                report = exact_match[0]
                logger.info(f"Exact name match found for {report.name}.")
                return self._format_result(
                    status="success",
                    data=json.loads(report.report_json),  # Changed from full_report_json
                    method="exact_name_match"
                )
            elif len(exact_match) > 1:
                names = [r.name for r in exact_match]
                logger.warning(f"Multiple exact name matches found for '{query}': {names}")
                return self._format_result(
                    status="multiple_matches",
                    error=f"Multiple patients found with the name '{query}'. Please ask the user to clarify or provide a patient ID.",
                    suggestions=names
                )

            # Fuzzy matching
            all_patients = session.query(PatientReport).all()
            
            fuzzy_matches = []
            for report in all_patients:
                score = self._fuzzy_match_score(query, report.name)
                if score >= 0.70:
                    fuzzy_matches.append((score, report))
            
            fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
            
            if fuzzy_matches:
                best_score, best_match = fuzzy_matches[0]
                
                best_matches = [m for s, m in fuzzy_matches if s == best_score]
                
                if len(best_matches) == 1:
                    logger.info(f"Fuzzy match found for {best_match.name} with score {best_score:.2f}.")
                    return self._format_result(
                        status="success",
                        data=json.loads(best_match.report_json),  # Changed from full_report_json
                        method="fuzzy_match",
                        similarity=round(best_score, 2)
                    )
                else:
                    names = [r.name for r in best_matches]
                    logger.warning(f"Multiple fuzzy matches found with score {best_score:.2f}: {names}")
                    return self._format_result(
                        status="multiple_matches",
                        error=f"Multiple close matches found for '{query}' (score {best_score:.2f}). Please ask the user to clarify.",
                        suggestions=names
                    )
            
            logger.warning(f"No patient found for query: '{query}'")
            return self._format_result(
                status="not_found",
                error=f"Patient with name or ID '{query}' could not be found. Please ask the user to confirm the spelling."
            )

        except Exception as e:
            logger.error(f"An unexpected database error occurred: {e}", exc_info=True)
            return self._format_result(
                status="error",
                error=f"An internal database error occurred: {e}"
            )
        finally:
            session.close()

def create_patient_retrieval_tool() -> Tool:
    patient_tool_instance = PatientDataTool()
    
    return Tool(
        name="patient_data_retrieval",
        description=(
            "A tool to retrieve a patient's full discharge report from the database. "
            "The input MUST be the patient's full name (e.g., 'Elias Vance') or their Patient ID (e.g., 'P001'). "
            "The tool handles fuzzy matching and returns a structured JSON object with the patient's data or an error/suggestion."
        ),
        func=patient_tool_instance.search_patients,
    )