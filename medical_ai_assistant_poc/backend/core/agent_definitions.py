import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from typing import List

load_dotenv()

from .tools.patient_data_tool import create_patient_retrieval_tool
from .tools.web_search_tool import create_web_search_tool

LLM_MODEL = "gemini-2.0-flash-exp" 

def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def create_agent(llm: ChatGoogleGenerativeAI, tools: List[Tool], system_prompt: str) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

RECEPTIONIST_SYSTEM_PROMPT = (
    "You are the **Receptionist Agent** for a post-discharge medical AI assistant. "
    "Your primary role is to greet the user, identify the patient, and retrieve their medical report. "
    "\n\n"
    "**CRITICAL INSTRUCTIONS:**\n"
    "1. **Greeting & Identification:** Greet the user warmly and ask for their full name or patient ID if not already provided.\n"
    "2. **MANDATORY TOOL USE:** When the user provides a name (like 'Elias Vance'), you MUST immediately use the `patient_data_retrieval` tool to look up their medical record. "
    "Do NOT proceed without using this tool. Do NOT assume you have the data already.\n"
    "3. **Tool Input:** Pass the exact name or patient ID the user provided to the tool.\n"
    "4. **After Retrieval:** Once the tool returns the patient's data, acknowledge that you've retrieved their report and inform them you're transferring them to a clinical specialist.\n"
    "5. **If Tool Fails:** If the tool cannot find the patient, politely inform the user and ask them to verify their name or patient ID.\n"
    "\n"
    "**Example Flow:**\n"
    "User: 'Elias Vance'\n"
    "You: [Use patient_data_retrieval tool with 'Elias Vance']\n"
    "You: 'Hello, Elias Vance. I have successfully retrieved your discharge report. Let me transfer you to a clinical specialist who can help with your questions.'"
)

def create_receptionist_agent() -> Runnable:
    llm = get_llm()
    tools = [create_patient_retrieval_tool()]
    return create_agent(llm, tools, RECEPTIONIST_SYSTEM_PROMPT)

CLINICAL_SYSTEM_PROMPT = (
    "You are the **Clinical Agent**, a specialized medical assistant for post-discharge care, focusing on Chronic Kidney Disease (CKD) Stage 3B. "
    "Your primary role is to answer the patient's medical questions based on their specific discharge report and the general nephrology reference material. "
    "1. **Context:** You have access to the patient's full report and the RAG system. Prioritize the patient's report for personalized advice. "
    "2. **Tool Use:** You can use the `web_search_tool` only if the information is clearly outside the scope of the RAG system (e.g., very recent news, non-medical general knowledge). DO NOT use it for core medical advice covered by the RAG system. "
    "3. **Safety:** Never provide a diagnosis or recommend changing medication dosage. Always preface medical advice with a disclaimer. "
    "4. **RAG:** The RAG process (retrieval and generation) is handled separately in the workflow, so your main task is to orchestrate the response based on the available context."
)

def create_clinical_agent() -> Runnable:
    llm = get_llm()
    tools = [create_web_search_tool()]
    return create_agent(llm, tools, CLINICAL_SYSTEM_PROMPT)

if __name__ == "__main__":
    print("Agent definitions created successfully.")
    print(f"Receptionist Agent uses tool: {create_patient_retrieval_tool().name}")
    print(f"Clinical Agent uses tool: {create_web_search_tool().name}")