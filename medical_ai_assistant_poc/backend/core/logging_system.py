import os
import json  # ✅ ADDED: Missing import
import logging
from typing import Any, Dict, List, Optional

# --- Configuration ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
MAIN_LOG_FILE = os.path.join(LOG_DIR, "assistant_workflow.log")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging System ---

def setup_logging(log_file: str = MAIN_LOG_FILE):
    """
    Sets up the main logging configuration for the application.
    """
    # Create a logger
    logger = logging.getLogger("assistant_logger")
    logger.setLevel(logging.INFO)
    
    # Prevent log messages from propagating to the root logger
    logger.propagate = False 

    # Check if handlers already exist to prevent duplicate logs
    if not logger.handlers:
        # Create file handler which logs even debug messages
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Initialize the main logger
logger = setup_logging()

def log_event(event_type: str, message: str, session_id: Optional[str] = None, details: Optional[dict] = None):
    """
    Logs a structured event to the main application log.
    
    :param event_type: A high-level category (e.g., "USER_INPUT", "AGENT_HANDOFF", "TOOL_CALL").
    :param message: A concise description of the event.
    :param session_id: The unique ID for the current user session.
    :param details: A dictionary of additional context to log.
    """
    log_message = f"[{event_type}]"
    if session_id:
        log_message += f" [Session:{session_id}]"
    log_message += f" {message}"
    
    if details:
        # Append details as a JSON string for easy parsing
        log_message += f" | Details: {json.dumps(details)}"
        
    logger.info(log_message)

# --- Agent and Workflow Specific Logging Functions ---

def log_user_input(session_id: str, user_message: str):
    log_event("USER_INPUT", "New message received from user.", session_id, {"message": user_message})

def log_agent_handoff(session_id: str, from_agent: str, to_agent: str, reason: str):
    log_event("AGENT_HANDOFF", f"Control transferred from {from_agent} to {to_agent}.", session_id, {"reason": reason})

def log_tool_call(session_id: str, tool_name: str, input_data: Any):
    log_event("TOOL_CALL", f"Tool '{tool_name}' called.", session_id, {"input": str(input_data)})

def log_tool_result(session_id: str, tool_name: str, result: Any):
    log_event("TOOL_RESULT", f"Tool '{tool_name}' returned a result.", session_id, {"result": str(result)})

def log_safety_check(session_id: str, classification: str, is_urgent: bool, reasoning: str):
    log_event("SAFETY_CHECK", f"Query classified as {classification}.", session_id, {
        "classification": classification,
        "is_urgent": is_urgent,
        "reasoning": reasoning
    })

def log_rag_result(session_id: str, query: str, num_documents: int, final_response_length: int):
    log_event("RAG_RESULT", "RAG generation complete.", session_id, {
        "query": query,
        "documents_retrieved": num_documents,
        "response_length": final_response_length
    })

def log_error(session_id: str, error_message: str, component: str):
    logger.error(f"[ERROR] [Session:{session_id}] Component: {component} | Message: {error_message}")

if __name__ == "__main__":
    # Example usage
    session_id = "session_123"
    log_user_input(session_id, "Hi, I'm John Smith and I have a question about my diet.")
    log_agent_handoff(session_id, "Receptionist", "Clinical", "Query is medical.")
    log_tool_call(session_id, "patient_data_retrieval", "John Smith")
    log_tool_result(session_id, "patient_data_retrieval", {"status": "success", "name": "John Smith"})
    log_safety_check(session_id, "Medical_Routine", False, "No warning signs mentioned.")
    log_rag_result(session_id, "diet question", 3, 550)
    log_error(session_id, "Database connection failed.", "PatientDataTool")
    
    print(f"\nCheck the log file at: {MAIN_LOG_FILE}")