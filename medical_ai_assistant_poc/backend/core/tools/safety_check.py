import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_MODEL = "gemini-2.0-flash-exp" 

# --- Pydantic Schema for Structured Output ---

class SafetyCheckResult(BaseModel):
    """Structured output for the safety and urgency check."""
    classification: str = Field(
        description="The classification of the user's query. Must be one of: 'Urgent', 'Medical_Routine', 'Administrative', or 'Harmful'."
    )
    urgency_reasoning: str = Field(
        description="A brief explanation of why the query was classified as Urgent, Harmful, or why it is safe."
    )
    is_urgent: bool = Field(
        description="True if the query is classified as 'Urgent' or 'Harmful', False otherwise."
    )

# --- Safety Check Logic ---

class SafetyChecker:
    """
    A tool to check the safety and urgency of a user's query, especially
    in the context of their specific medical condition and warning signs.
    """
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        """Creates the prompt template for the safety check."""
        template = (
            "You are a highly specialized medical safety classification agent. Your sole task is to analyze a user's query "
            "in the context of their medical report and determine its safety and urgency.\n\n"
            "You MUST respond with ONLY a valid JSON object with these exact fields:\n"
            '{{\"classification\": \"<value>\", \"urgency_reasoning\": \"<explanation>\", \"is_urgent\": <true/false>}}\n\n'
            "--- PATIENT REPORT ---\n"
            "{patient_report}\n\n"
            "--- USER QUERY ---\n"
            "{query}\n\n"
            "Based on the patient's report (especially the 'warning_signs') and the user's query, classify the query.\n\n"
            "Classification rules:\n"
            "1. 'Urgent': If the query mentions a symptom that is a 'warning_sign' in the patient's report (e.g., 'shortness of breath', 'rapid weight gain', 'chest pain') or suggests an immediate, life-threatening issue.\n"
            "2. 'Harmful': If the query is inappropriate, malicious, or asks for dangerous medical advice (e.g., 'how to overdose').\n"
            "3. 'Administrative': If the query is ONLY a patient name, patient ID, greeting, or about appointments/billing/system usage. Examples: 'Elias Vance', 'P001', 'Hello', 'How do I use this?'\n"
            "4. 'Medical_Routine': All other medical questions that are not urgent (e.g., 'what is my dosage', 'what foods should I avoid').\n\n"
            "**CRITICAL**: If the query is JUST a name or patient ID with no medical question (like 'Elias Vance', 'John Smith', 'P001'), "
            "you MUST classify it as 'Administrative' with is_urgent: false. This is just identification, not a medical emergency.\n\n"
            "Pay close attention to the patient's specific warning signs.\n\n"
            "Respond with ONLY the JSON object, no other text."
        )
        return ChatPromptTemplate.from_template(template)

    def check_safety(self, query: str, patient_report: str) -> Dict[str, Any]:
        """
        Performs the safety and urgency check.
        
        :param query: The user's question.
        :param patient_report: The patient's full report as a JSON string.
        :return: The structured safety check result as a dictionary.
        """
        try:
            # Format the prompt
            messages = self.prompt.format_messages(
                query=query,
                patient_report=patient_report
            )
            
            # Invoke the LLM
            response = self.llm.invoke(messages)
            
            # Get the response text
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["classification", "urgency_reasoning", "is_urgent"]
            if not all(field in result for field in required_fields):
                raise ValueError(f"Missing required fields. Got: {result.keys()}")
            
            # Ensure is_urgent is a boolean
            if isinstance(result["is_urgent"], str):
                result["is_urgent"] = result["is_urgent"].lower() == "true"
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
            # Default to safe classification on parsing error
            return {
                "classification": "Medical_Routine",
                "urgency_reasoning": f"JSON parsing failed, defaulting to safe classification: {e}",
                "is_urgent": False
            }
        except Exception as e:
            print(f"❌ Safety check error: {e}")
            import traceback
            traceback.print_exc()
            # Default to safe classification on error
            return {
                "classification": "Medical_Routine",
                "urgency_reasoning": f"LLM classification failed, defaulting to safe classification: {e}",
                "is_urgent": False
            }

def create_safety_check_tool() -> Tool:
    """
    Creates a LangChain Tool object that wraps the SafetyChecker's check_safety method.
    """
    safety_checker_instance = SafetyChecker()
    
    return Tool(
        name="safety_check_tool",
        description=(
            "A critical tool to classify the user's query for safety and urgency. "
            "Input MUST be a JSON string with two keys: 'query' (the user's message) and 'patient_report' (the patient's full report as a JSON string). "
            "The tool returns a structured JSON object with the classification and urgency status."
        ),
        func=safety_checker_instance.check_safety,
    )

if __name__ == "__main__":
    # Example usage
    checker = SafetyChecker()
    dummy_report = json.dumps({
        "patient_id": "P001",
        "name": "Elias Vance",
        "diagnosis": "Chronic Kidney Disease (CKD) Stage 3B",
        "warning_signs": ["Swelling in legs or ankles", "Shortness of breath", "Rapid weight gain (> 2 lbs/day)", "Severe fatigue"]
    })
    
    # Test 1: Just a name (should be Administrative)
    name_query = "Elias Vance"
    print(f"Test 1 - Name only: '{name_query}'")
    result = checker.check_safety(name_query, dummy_report)
    print(json.dumps(result, indent=2))
    print()
    
    # Test 2: Urgent query
    urgent_query = "I woke up and I can't catch my breath, and my ankles are huge."
    print(f"Test 2 - Urgent query: '{urgent_query}'")
    result = checker.check_safety(urgent_query, dummy_report)
    print(json.dumps(result, indent=2))
    print()
    
    # Test 3: Routine query
    routine_query = "What is the dosage for my Lisinopril?"
    print(f"Test 3 - Routine query: '{routine_query}'")
    result = checker.check_safety(routine_query, dummy_report)
    print(json.dumps(result, indent=2))