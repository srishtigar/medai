import operator
import json
import sqlite3
import os
from typing import TypedDict, Annotated, List, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

# Import core components
from .agent_definitions import create_receptionist_agent, create_clinical_agent
from .tools.patient_data_tool import PatientDataTool
from .tools.safety_check import SafetyChecker
from .rag_pipeline.retriever import rag_retriever
from .rag_pipeline.generator import rag_generator

from .logging_system import (
    log_user_input, log_agent_handoff, log_tool_call, log_tool_result, 
    log_safety_check, log_rag_result, log_error
)

# --- 1. Define Graph State ---

class AgentState(TypedDict):
    """
    Represents the state of the LangGraph workflow.
    """
    # User input
    input: str
    # History of messages
    messages: Annotated[List[BaseMessage], operator.add]
    chat_history: Annotated[List[BaseMessage], operator.add]
    # Current agent's response (or tool call)
    agent_output: Any
    # Unique session ID for logging
    session_id: str
    
    # Patient Data
    patient_id: str
    patient_name: str
    patient_report: str # Full JSON string of the patient report
    
    # Flag to track if this is the first interaction after patient identification
    patient_just_identified: bool
    
    # Safety Check Result
    safety_check_result: dict
    
    # RAG Context
    rag_context: List[Document]
    
    # Final response to user
    final_response: str

# --- 2. Define Nodes (Functions) ---

# Initialize Agents and Tools
receptionist_agent = create_receptionist_agent()
clinical_agent = create_clinical_agent()
safety_checker = SafetyChecker()

def call_receptionist(state: AgentState) -> AgentState:
    """Handles the initial greeting and patient identification."""
    session_id = state["session_id"]
    user_input = state["input"]
    
    log_user_input(session_id, user_input)
    log_agent_handoff(session_id, "START", "Receptionist", "Initial query.")
    
    # Get existing chat_history or initialize empty list
    chat_history = state.get("chat_history", [])
    
    # Create input dict for the agent
    agent_input = {
        "input": user_input,
        "chat_history": chat_history
    }
    
    try:
        print(f"🔵 DEBUG [Receptionist]: Calling agent with input: '{user_input[:100]}...'")
        
        # The receptionist agent will use the patient_data_retrieval tool
        result = receptionist_agent.invoke(agent_input)
        
        print(f"🔵 DEBUG [Receptionist]: Result type: {type(result)}")
        print(f"🔵 DEBUG [Receptionist]: Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        if isinstance(result, dict) and "output" in result:
            print(f"🔵 DEBUG [Receptionist]: Output preview: {result['output'][:100]}...")
        
        # Add messages to chat history for next iteration
        chat_history.append(HumanMessage(content=user_input))
        
        # Extract the agent's response
        if isinstance(result, dict) and "output" in result:
            chat_history.append(AIMessage(content=result["output"]))
        
        return {
            "agent_output": result,
            "chat_history": chat_history,
            "messages": [HumanMessage(content=user_input)]
        }
    except Exception as e:
        print(f"❌ ERROR [Receptionist]: {str(e)}")
        log_error(session_id, f"Error in receptionist agent: {str(e)}", "call_receptionist")
        return {
            "agent_output": {"error": str(e)},
            "chat_history": chat_history,
            "messages": [HumanMessage(content=user_input)]
        }

def handle_patient_retrieval(state: AgentState) -> AgentState:
    """Processes the result of the patient data retrieval tool call."""
    session_id = state["session_id"]
    agent_output = state["agent_output"]
    
    print(f"🟡 DEBUG [Handle Retrieval]: Starting...")
    print(f"🟡 DEBUG [Handle Retrieval]: agent_output type: {type(agent_output)}")
    
    # Handle error from previous node
    if isinstance(agent_output, dict) and "error" in agent_output:
        error_msg = f"I encountered an error: {agent_output['error']}. Please try again."
        print(f"🟡 DEBUG [Handle Retrieval]: Returning error message")
        return {"agent_output": error_msg, "patient_just_identified": False}
    
    # Check if the agent successfully called the tool
    if isinstance(agent_output, dict) and "intermediate_steps" in agent_output:
        print(f"🟡 DEBUG [Handle Retrieval]: Found intermediate_steps, checking for tool calls...")
        
        for step in agent_output["intermediate_steps"]:
            action, tool_result = step
            
            print(f"🟡 DEBUG [Handle Retrieval]: Tool called: {action.tool}")
            
            if action.tool == "patient_data_retrieval":
                log_tool_call(session_id, "patient_data_retrieval", {"query": action.tool_input})
                
                if isinstance(tool_result, dict):
                    log_tool_result(session_id, "patient_data_retrieval", tool_result)
                    
                    print(f"🟡 DEBUG [Handle Retrieval]: Tool result status: {tool_result.get('status')}")
                    
                    if tool_result.get("status") == "success":
                        patient_data = tool_result["data"]
                        print(f"🟡 DEBUG [Handle Retrieval]: ✅ Patient found: {patient_data['name']}")
                        
                        # Generate a friendly follow-up message
                        patient_name = patient_data['name']
                        diagnosis = patient_data.get('diagnosis', 'your condition')
                        
                        follow_up_message = (
                            f"Hello, {patient_name}! I've retrieved your discharge report from {patient_data.get('discharge_date', 'your recent discharge')} "
                            f"for {diagnosis}. How are you feeling today? Do you have any questions about your medications, "
                            f"dietary restrictions, or any symptoms you're experiencing?"
                        )
                        
                        return {
                            "patient_id": patient_data["patient_id"],
                            "patient_name": patient_data["name"],
                            "patient_report": json.dumps(patient_data),
                            "patient_just_identified": True,  # Flag that patient was just identified
                            "agent_output": follow_up_message,
                            "final_response": follow_up_message  # Set as final response to return to user
                        }
                    elif tool_result.get("status") == "multiple_matches":
                        print(f"🟡 DEBUG [Handle Retrieval]: Multiple matches found")
                        return {
                            "agent_output": tool_result["error"],
                            "final_response": tool_result["error"],
                            "patient_just_identified": False
                        }
                    else:
                        print(f"🟡 DEBUG [Handle Retrieval]: Patient not found")
                        error_msg = tool_result.get("error", "Patient not found.")
                        return {
                            "agent_output": error_msg,
                            "final_response": error_msg,
                            "patient_just_identified": False
                        }
    
    # If no tool was called, return the agent's output message
    if isinstance(agent_output, dict) and "output" in agent_output:
        print(f"🟡 DEBUG [Handle Retrieval]: No tool called, returning agent output")
        return {
            "agent_output": agent_output["output"],
            "final_response": agent_output["output"],
            "patient_just_identified": False
        }
    
    # Fallback
    print(f"🟡 DEBUG [Handle Retrieval]: Fallback - converting to string")
    output_str = str(agent_output)
    return {
        "agent_output": output_str,
        "final_response": output_str,
        "patient_just_identified": False
    }

def check_safety_node(state: AgentState) -> AgentState:
    """Performs the critical safety and urgency check."""
    session_id = state["session_id"]
    
    print(f"🟠 DEBUG [Safety Check]: Starting...")
    
    # Only proceed if patient data is available
    if not state.get("patient_report"):
        print(f"🟠 DEBUG [Safety Check]: ERROR - No patient report found!")
        log_error(session_id, "Safety check called without patient report.", "SafetyCheckNode")
        return {
            "safety_check_result": {
                "is_urgent": False,
                "classification": "Administrative", 
                "urgency_reasoning": "No patient data available yet."
            }
        }

    # The query is the last user input
    query = state["input"]
    patient_report = state["patient_report"]
    
    print(f"🟠 DEBUG [Safety Check]: Checking query: '{query[:100]}...'")
    
    log_tool_call(session_id, "safety_check_tool", {"query": query, "patient_report": patient_report})
    
    try:
        # Invoke the safety checker
        result = safety_checker.check_safety(query, patient_report)
        
        print(f"🟠 DEBUG [Safety Check]: Result - Urgent: {result['is_urgent']}, Classification: {result['classification']}")
        
        log_safety_check(session_id, result["classification"], result["is_urgent"], result["urgency_reasoning"])
        
        return {"safety_check_result": result}
    except Exception as e:
        print(f"❌ ERROR [Safety Check]: {str(e)}")
        log_error(session_id, f"Error in safety check: {str(e)}", "check_safety_node")
        return {
            "safety_check_result": {
                "is_urgent": False,
                "classification": "Medical_Routine",
                "urgency_reasoning": "Safety check error, defaulting to safe."
            }
        }

def call_clinical(state: AgentState) -> AgentState:
    """Handles the RAG process and final answer generation."""
    session_id = state["session_id"]
    log_agent_handoff(session_id, "Receptionist/SafetyCheck", "Clinical", "Query is safe and medical.")
    
    print(f"🔵 DEBUG [Clinical]: Starting RAG process...")
    
    query = state["input"]
    patient_report = state["patient_report"]
    
    try:
        # 1. RAG Retrieval
        print(f"🔵 DEBUG [Clinical]: Retrieving context...")
        rag_context = rag_retriever.retrieve(query)
        print(f"🔵 DEBUG [Clinical]: Retrieved {len(rag_context)} documents")
        
        # 2. RAG Generation
        print(f"🔵 DEBUG [Clinical]: Generating response...")
        final_response = rag_generator.generate(query, rag_context, patient_report)
        print(f"🔵 DEBUG [Clinical]: Generated response length: {len(final_response)}")
        
        log_rag_result(session_id, query, len(rag_context), len(final_response))
        
        return {"final_response": final_response}
    except Exception as e:
        print(f"❌ ERROR [Clinical]: {str(e)}")
        log_error(session_id, f"Error in clinical agent: {str(e)}", "call_clinical")
        return {
            "final_response": "I apologize, but I encountered an error processing your request. Please try rephrasing your question."
        }

def emergency_exit(state: AgentState) -> AgentState:
    """Returns a hard-coded safety message for urgent/harmful queries."""
    session_id = state["session_id"]
    log_agent_handoff(session_id, "SafetyCheck", "EmergencyExit", "Query classified as urgent/harmful.")
    
    print(f"🚨 DEBUG [Emergency Exit]: Triggered!")
    
    safety_message = (
        "**EMERGENCY ALERT:** This is an emergency. Based on your input, you may be experiencing a critical medical situation. "
        "**Please contact 911 or your healthcare provider immediately.** "
        "This AI assistant cannot provide emergency medical care."
    )
    return {"final_response": safety_message}

def format_output(state: AgentState) -> AgentState:
    """Adds the required medical disclaimers to the final response."""
    
    response = state.get("final_response", "")
    
    print(f"🟢 DEBUG [Format Output]: Starting...")
    print(f"🟢 DEBUG [Format Output]: Input response length: {len(response)}")
    print(f"🟢 DEBUG [Format Output]: Response preview: {response[:200]}...")
    
    # If no final_response is set, this is an error
    if not response:
        print(f"⚠️  WARNING [Format Output]: Empty final_response!")
        response = "I apologize, but I couldn't generate a response. Please try again."
    
    # Add mandatory disclaimers
    disclaimer = (
        "\n\n---\n"
        "**Disclaimer:** This is an AI assistant for educational and informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult healthcare professionals for medical advice."
    )
    
    final = response + disclaimer
    print(f"🟢 DEBUG [Format Output]: Final response length: {len(final)}")
    
    return {"final_response": final}

# --- 3. Define Conditional Edges (Routing) ---

def route_after_retrieval(state: AgentState) -> str:
    """Routes the flow after patient retrieval."""
    
    print(f"🔀 DEBUG [Route After Retrieval]: Starting...")
    print(f"🔀 DEBUG [Route]: patient_report exists: {bool(state.get('patient_report'))}")
    print(f"🔀 DEBUG [Route]: patient_just_identified: {state.get('patient_just_identified', False)}")
    
    # If patient was just identified, END here and wait for next user message
    if state.get("patient_just_identified", False):
        print(f"🔀 DEBUG [Route]: ✅ Patient just identified → Going to format_output (END after)")
        # Reset the flag for next iteration
        state["patient_just_identified"] = False
        return "format_output"
    
    # If patient already identified and this is a new query, go to safety check
    if state.get("patient_report"):
        print(f"🔀 DEBUG [Route]: ✅ Patient already identified, new query → Going to safety_check")
        return "safety_check"
    
    # No patient data, return response and end
    print(f"🔀 DEBUG [Route]: ❌ No patient data → Going to format_output")
    return "format_output"

def route_after_safety_check(state: AgentState) -> str:
    """Routes the flow based on the safety check result."""
    result = state.get("safety_check_result", {})
    
    print(f"🔀 DEBUG [Route After Safety]: Starting...")
    print(f"🔀 DEBUG [Route After Safety]: is_urgent: {result.get('is_urgent', False)}")
    print(f"🔀 DEBUG [Route After Safety]: classification: {result.get('classification', 'N/A')}")
    
    if result.get("is_urgent", False):
        print(f"🔀 DEBUG [Route After Safety]: → Going to emergency_exit")
        return "emergency_exit"
    
    # If not urgent, proceed to the clinical agent for RAG
    print(f"🔀 DEBUG [Route After Safety]: → Going to clinical_agent")
    return "clinical_agent"

# --- 4. Build the Graph ---

def build_workflow() -> StateGraph:
    """Builds and compiles the LangGraph workflow."""
    
    print("🏗️  Building LangGraph workflow...")
    
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    print("  📍 Adding nodes...")
    workflow.add_node("receptionist", call_receptionist)
    workflow.add_node("handle_retrieval", handle_patient_retrieval)
    workflow.add_node("safety_check", check_safety_node)
    workflow.add_node("clinical_agent", call_clinical)
    workflow.add_node("emergency_exit", emergency_exit)
    workflow.add_node("format_output", format_output)
    
    # Set Entry Point
    print("  🚪 Setting entry point...")
    workflow.set_entry_point("receptionist")
    
    # Define Edges
    print("  🔗 Adding edges...")
    
    # 1. Receptionist -> Handle Retrieval
    workflow.add_edge("receptionist", "handle_retrieval")
    
    # 2. Handle Retrieval -> Conditional Routing
    # CRITICAL: If patient just identified, go to format_output and END
    # If patient already exists, go to safety_check
    workflow.add_conditional_edges(
        "handle_retrieval",
        route_after_retrieval,
        {
            "format_output": "format_output",  # Patient just identified OR error
            "safety_check": "safety_check"      # Patient already exists, new medical query
        }
    )
    
    # 3. Safety Check -> Conditional Routing
    workflow.add_conditional_edges(
        "safety_check",
        route_after_safety_check,
        {
            "emergency_exit": "emergency_exit",
            "clinical_agent": "clinical_agent"
        }
    )
    
    # 4. Clinical Agent -> Format Output
    workflow.add_edge("clinical_agent", "format_output")
    
    # 5. Emergency Exit -> Format Output
    workflow.add_edge("emergency_exit", "format_output")
    
    # 6. Format Output -> END
    workflow.add_edge("format_output", END)
    
    # CRITICAL FIX: Create SQLite connection for checkpointing (PERSISTENT FILE)
    print("  💾 Setting up checkpointer...")
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the backend directory
    backend_dir = os.path.dirname(os.path.dirname(current_dir))
    # Create checkpoints.db in the backend directory
    db_path = os.path.join(backend_dir, "checkpoints.db")
    print(f"  💾 Checkpoint database: {db_path}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create persistent SQLite connection
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # Compile the graph
    print("  ⚙️  Compiling workflow...")
    app = workflow.compile(
        checkpointer=SqliteSaver(conn),
        debug=False
    )
    
    print("✅ Workflow built successfully!")
    print(f"✅ Checkpoints will be saved to: {db_path}")
    return app

# --- 5. Main Function for FastAPI Integration ---

def get_workflow_app():
    """Returns the compiled LangGraph application."""
    return build_workflow()

if __name__ == "__main__":
    print("🧪 Testing workflow...")
    app = build_workflow()
    print("✅ Workflow test complete!")