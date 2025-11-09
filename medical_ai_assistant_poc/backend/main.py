import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import HumanMessage, AIMessage
from typing import List

# Import the workflow builder
from core.langgraph_workflow import get_workflow_app, AgentState
from core.logging_system import log_user_input, log_error

# --- Configuration ---
app = FastAPI(
    title="Medical AI Assistant Backend",
    description="FastAPI application serving the LangGraph-based medical assistant workflow.",
    version="1.0.0"
)

# Initialize the LangGraph application
# Note: The SqliteSaver is initialized within the workflow builder in langgraph_workflow.py
# The checkpointer will save state to a file named 'checkpoints.db' in the current directory.
print("🚀 Initializing workflow...")
app_workflow = get_workflow_app()
print("✅ Workflow initialized successfully!")

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    """Model for the incoming chat message."""
    session_id: str
    message: str

class ChatResponse(BaseModel):
    """Model for the outgoing chat response."""
    session_id: str
    response: str
    chat_history: List[dict]

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Processes a user message through the LangGraph workflow.
    """
    session_id = request.session_id
    user_message = request.message
    
    print(f"\n{'='*80}")
    print(f"📨 NEW REQUEST | Session: {session_id[:8]}... | Message: '{user_message[:50]}...'")
    print(f"{'='*80}\n")
    
    log_user_input(session_id, user_message)
    
    # Configuration for checkpointer - this is how state is restored
    config = {"configurable": {"thread_id": session_id}}
    
    # CRITICAL FIX: Check if state exists in checkpointer
    # If patient data already exists, don't reset it
    try:
        current_state = app_workflow.get_state(config)
        
        print(f"🔍 DEBUG [Main]: Checking for existing state...")
        
        if current_state and current_state.values:
            print(f"🔍 DEBUG [Main]: ✅ Found existing state")
            print(f"🔍 DEBUG [Main]: State keys: {list(current_state.values.keys())}")
            
            # Check if patient data exists
            has_patient = bool(current_state.values.get("patient_report"))
            print(f"🔍 DEBUG [Main]: Has patient data: {has_patient}")
            
            if has_patient:
                print(f"🔍 DEBUG [Main]: Patient: {current_state.values.get('patient_name', 'Unknown')}")
                print(f"🔍 DEBUG [Main]: 🎯 PRESERVING patient data for follow-up question")
                
                # IMPORTANT: Only pass NEW input, checkpointer will restore the rest
                inputs = {
                    "input": user_message,
                    "session_id": session_id,
                    # Don't initialize these - let checkpointer restore them
                    # This is the key to preserving patient data!
                }
            else:
                print(f"🔍 DEBUG [Main]: No patient data yet, initializing fresh state")
                inputs = {
                    "input": user_message,
                    "session_id": session_id,
                    "messages": [],
                    "chat_history": [],
                    "agent_output": None,
                    "patient_id": "",
                    "patient_name": "",
                    "patient_report": "",
                    "patient_just_identified": False,
                    "safety_check_result": {},
                    "rag_context": [],
                    "final_response": ""
                }
        else:
            print(f"🔍 DEBUG [Main]: 🆕 No existing state (first message)")
            # First message in this session
            inputs = {
                "input": user_message,
                "session_id": session_id,
                "messages": [],
                "chat_history": [],
                "agent_output": None,
                "patient_id": "",
                "patient_name": "",
                "patient_report": "",
                "patient_just_identified": False,
                "safety_check_result": {},
                "rag_context": [],
                "final_response": ""
            }
    except Exception as e:
        # Checkpointer error or first message
        print(f"ℹ️  DEBUG [Main]: Could not retrieve state (likely first message): {str(e)[:100]}")
        inputs = {
            "input": user_message,
            "session_id": session_id,
            "messages": [],
            "chat_history": [],
            "agent_output": None,
            "patient_id": "",
            "patient_name": "",
            "patient_report": "",
            "patient_just_identified": False,
            "safety_check_result": {},
            "rag_context": [],
            "final_response": ""
        }
    
    print(f"🔧 DEBUG [Main]: Input state keys: {list(inputs.keys())}")
    
    try:
        # 2. Invoke the workflow with config
        # The checkpointer will merge our input with the stored state
        print(f"🔄 DEBUG [Main]: Invoking workflow with session: {session_id[:8]}...")
        
        final_state = app_workflow.invoke(inputs, config=config)
        
        print(f"✅ DEBUG [Main]: Workflow completed!")
        print(f"🔍 DEBUG [Main]: Final state keys: {list(final_state.keys())}")
        print(f"🔍 DEBUG [Main]: Final patient_report exists: {bool(final_state.get('patient_report'))}")
        
        # 3. Extract the final response
        response_text = final_state.get("final_response", "")
        
        print(f"📤 DEBUG [Main]: final_response exists: {bool(response_text)}")
        print(f"📤 DEBUG [Main]: final_response length: {len(response_text)}")
        print(f"📤 DEBUG [Main]: final_response preview: {response_text[:200]}...")
        
        # Fallback if no response was generated
        if not response_text:
            print(f"⚠️  WARNING [Main]: Empty final_response! Using fallback message.")
            response_text = "I apologize, but I couldn't generate a response. Please try again."
        
        # 4. Update chat history for the response
        dummy_history = [
            {"type": "human", "content": user_message},
            {"type": "ai", "content": response_text}
        ]
        
        print(f"\n{'='*80}")
        print(f"✅ RESPONSE SENT | Length: {len(response_text)} chars")
        print(f"{'='*80}\n")
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            chat_history=dummy_history
        )
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR IN CHAT ENDPOINT")
        print(f"{'='*80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        import traceback
        print(f"\n🔍 Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}\n")
        
        log_error(session_id, str(e), "FastAPI Chat Endpoint")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/new_session")
async def new_session():
    """
    Generates a new unique session ID for a new user chat.
    """
    new_id = str(uuid.uuid4())
    print(f"🆕 New session created: {new_id}")
    return {"session_id": new_id}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Medical AI Assistant Backend"}

@app.get("/")
async def root():
    return {"message": "Medical AI Assistant Backend is running."}

# Add startup event to verify everything is loaded
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    print("\n" + "="*80)
    print("🏥 MEDICAL AI ASSISTANT BACKEND")
    print("="*80)
    print("✅ FastAPI server started")
    print("✅ Workflow loaded")
    print("✅ Ready to accept requests")
    
    # Check if checkpoints.db exists
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints.db")
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint database found: {checkpoint_path}")
    else:
        print(f"ℹ️  Checkpoint database will be created on first use")
    
    print("="*80 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    print("\n" + "="*80)
    print("🛑 MEDICAL AI ASSISTANT BACKEND SHUTTING DOWN")
    print("="*80 + "\n")

if __name__ == "__main__":
    import uvicorn
    # Ensure the logs directory exists before starting
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"📁 Logs directory: {logs_dir}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)