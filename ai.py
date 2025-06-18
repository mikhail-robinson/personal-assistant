import datetime
import os

import streamlit as st
from langchain_core.messages import \
    HumanMessage  # For run_gsuite_agent and central agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from mcp_use import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter


def create_llm():
    """Create a language model instance"""

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY not found in environment variables. Please add it to your .env file."
        )
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)

    return llm


def create_agent_prompt():
    today = datetime.datetime.today()
    system_message_content = (
    f"IMPORTANT: The current date and time is {today}. You are a specialized GSuite assistant. "
    "Your task is to process requests related to Gmail and Google Calendar and return ONLY the direct factual answer or a summary of the action taken. "
    "Do not include conversational phrases, acknowledgments of tool use, or any other text beyond the direct result. "
    "For example, if asked for calendar events, return only the event details. If asked to read an email, return only the email content."
)

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


async def google_mcp_tools():
    client = MCPClient.from_config_file("mcp_config.json")
    adapter = LangChainAdapter()
    try:
        tools = await adapter.create_tools(client)
        st.sidebar.success(
            f"✅ {len(tools)} LangChain tool(s) created: {[tool.name for tool in tools]}."
        )
    except Exception as e:
        st.sidebar.error(f"❌ Failed to create LangChain tools: {str(e)}")
        return None
    if tools:
        return tools


async def create_gsuite_agent_executor(): # Renamed
    llm = create_llm()
    if not llm:
        st.sidebar.error("❌ GSuite Agent: Failed to create language model.")
        return None
    prompt = create_agent_prompt() # This prompt is now tailored for GSuite agent
    tools = await google_mcp_tools()
    if not tools:
        st.sidebar.error("❌ GSuite Agent: MCP tools failed to load for GSuite agent.")
        return None

    agent = create_react_agent(llm, tools, prompt=prompt)

    st.sidebar.success("✅ GSuite AgentExecutor ready!") # Updated message
    return agent


# --- Phase 1: GSuite Agent as a Tool ---
async def run_gsuite_agent(query: str) -> str:
    """Runs the GSuite agent with the given query and returns its output."""
    gsuite_agent_executor = await create_gsuite_agent_executor()
    if not gsuite_agent_executor:
        return "Error: GSuite agent failed to initialize."

    try:
        # Construct the messages input for the GSuite agent
        # The ReAct agent created by create_react_agent expects a 'messages' list.
        print(f"[run_gsuite_agent] Invoking GSuite agent with query: '{query}'")
        response = await gsuite_agent_executor.ainvoke({"messages": [HumanMessage(content=query)]})
        print(f"[run_gsuite_agent] Raw response from GSuite agent: {response}")

        # Extracting output from a ReAct agent's ainvoke can be tricky.
        # It often returns a dict with 'output' or the last message in a 'messages' list.
        final_output = "GSuite agent did not provide a recognizable output."
        if isinstance(response, dict):
            if "output" in response:
                final_output = response["output"]
            elif "messages" in response and isinstance(response["messages"], list) and response["messages"]:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    final_output = last_message.content

        # If response itself is a string (can happen with some agent configurations)
        elif isinstance(response, str):
            final_output = response

        if not isinstance(final_output, str):
            final_output = str(final_output) # Ensure it's a string

        print(f"[run_gsuite_agent] Extracted final_output to be returned: '{final_output}'")
        return final_output
    except Exception as e:
        # Log the error for debugging if possible, e.g., using st.exception(e) or logging module
        st.error(f"Error running GSuite agent: {e}")
        return f"Error processing your GSuite request: {str(e)}"

gsuite_tool = Tool(
    name="GSuiteAssistant",
    func=None,
    coroutine=run_gsuite_agent,
    description="Use this tool for any questions or tasks specifically related to accessing or managing GSuite services like Gmail and Google Calendar. Input should be the original user query about email or calendar.",
)


# --- Phase 2: Central LLM Agent ---
def create_central_llm_prompt():
    """Creates the prompt for the Central LLM Agent."""
    today = datetime.datetime.today()
    system_message_content = (
        f"IMPORTANT: The current date and time is {today}. You are a helpful general-purpose AI assistant.\n"
        "You have access to a specialized tool for GSuite (Gmail and Google Calendar). When a user's query requires this capability,\n"
        "FIRST, provide a brief textual acknowledgment of your intention to use the tool (e.g., 'Okay, I'll use my GSuite assistant for that.').\n"
        "THEN, use the GSuiteAssistant tool. After the tool provides its response, present that information clearly to the user.\n"
        "If the query is general and doesn't require the GSuite tool, answer it directly using your knowledge."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

async def create_central_llm_agent():
    """Creates the Central LLM Agent."""
    llm = create_llm()
    if not llm:
        st.error("❌ Central LLM: Failed to create language model.")
        return None

    prompt = create_central_llm_prompt()
    tools_list = [gsuite_tool] # Only GSuite tool for now

    # Ensure the input key for create_react_agent matches MessagesPlaceholder, typically "messages"
    central_agent_executor = create_react_agent(llm, tools_list, prompt=prompt)

    st.sidebar.success("✅ Central LLM Agent ready!")
    return central_agent_executor
