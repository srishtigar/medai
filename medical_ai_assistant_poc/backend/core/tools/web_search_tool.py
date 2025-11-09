import os
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# --- Web Search Tool Logic ---

def create_web_search_tool() -> Tool:
    """
    Creates a LangChain Tool object that wraps a general web search engine.
    We use DuckDuckGoSearchRun as a simple, readily available search tool.
    """
    # Initialize the search tool
    search = DuckDuckGoSearchRun()
    
    # Wrap it in a LangChain Tool object
    return Tool(
        name="web_search_tool",
        description=(
            "A tool for performing general web searches to find up-to-date or external information "
            "that is not available in the internal nephrology reference document. "
            "Input MUST be a concise, single-string search query (e.g., 'new guidelines for CKD diet')."
        ),
        func=search.run,
    )

if __name__ == "__main__":
    # Example usage
    tool = create_web_search_tool()
    query = "latest research on kidney disease and diet"
    print(f"Searching for: {query}")
    
    # Note: Running this will execute the search and may take a moment.
    # result = tool.run(query)
    # print("\n--- SEARCH RESULT (Snippet) ---")
    # print(result[:500] + "...")
    print("Example tool created. To run, uncomment the lines above.")
