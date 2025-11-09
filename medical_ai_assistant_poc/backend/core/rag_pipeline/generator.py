import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List

# Load environment variables
load_dotenv()

# --- Configuration ---
# Use a fast and capable model for generation
LLM_MODEL = "gemini-2.0-flash-exp"  
# --- RAG Generator Logic ---

class RAGGenerator:
    """
    Handles the final answer generation using the LLM, retrieved context, and user query.
    """
    def __init__(self):
        print("Initializing RAGGenerator...")
        # Use ChatGoogleGenerativeAI for Gemini models
        # Make sure GOOGLE_API_KEY is set in your .env file
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")  # ✅ Use Gemini API key
        )
        self.prompt = self._create_prompt()
        print("RAGGenerator initialized.")

    def _create_prompt(self) -> ChatPromptTemplate:
        """Creates the prompt template for the RAG generation."""
        template = (
            "You are a helpful medical assistant specializing in post-discharge care for Chronic Kidney Disease (CKD) patients. "
            "Your task is to answer the user's question based ONLY on the provided context and the patient's specific report. "
            "If the context or patient report does not contain the answer, state clearly that you cannot answer based on the provided information. "
            "Always maintain a professional, empathetic, and cautious tone. "
            "Cite the source of the information from the context if possible.\n\n"
            "--- PATIENT REPORT ---\n"
            "{patient_report}\n\n"
            "--- CONTEXT ---\n"
            "{context}\n\n"
            "--- QUESTION ---\n"
            "{question}"
        )
        return ChatPromptTemplate.from_template(template)

    def generate(self, question: str, context: List[Document], patient_report: str) -> str:
        """
        Generates the final response.
        
        :param question: The user's question.
        :param context: A list of relevant Document objects from the retriever.
        :param patient_report: The patient's full report as a string (or a summary).
        :return: The generated response text.
        """
        # Format the context for the prompt
        context_text = "\n---\n".join([doc.page_content for doc in context])
        
        # Create the chain
        rag_chain = self.prompt | self.llm
        
        # Invoke the chain
        response = rag_chain.invoke({
            "question": question,
            "context": context_text,
            "patient_report": patient_report
        })
        
        return response.content

# Global instance for easy access
rag_generator = RAGGenerator()

if __name__ == "__main__":
    # Example usage (requires a dummy patient report and context)
    dummy_patient_report = json.dumps({
        "name": "Test Patient",
        "diagnosis": "CKD Stage 3B",
        "warning_signs": ["Swelling in legs or ankles"]
    }, indent=2)
    
    dummy_context = [
        Document(page_content="CKD Stage 3B is defined by eGFR between 30 and 44 mL/min/1.73m²."),
        Document(page_content="Warning signs like swelling in legs or ankles should be monitored closely.")
    ]
    
    question = "What is CKD Stage 3B and what should I do about my swelling?"
    
    print(f"Question: {question}")
    response = rag_generator.generate(question, dummy_context, dummy_patient_report)
    print("\n--- GENERATED RESPONSE ---")
    print(response)