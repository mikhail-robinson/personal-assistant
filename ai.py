import datetime
import logging
import os

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from mcp_use import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter


def create_llm():
    """Create a language model instance"""

    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. Please add it to your .env file."
        )

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
        logging.info(
            f"✅ {len(tools)} LangChain tool(s) created: {[tool.name for tool in tools]}."
        )
    except Exception as e:
        raise RuntimeError(f"❌ Failed to create LangChain tools: {str(e)}")
    if tools:
        return tools


async def create_gsuite_agent_executor():
    llm = create_llm()
    prompt = create_agent_prompt()
    tools = await google_mcp_tools()
    agent = create_react_agent(llm, tools, prompt=prompt)

    logging.info("✅ GSuite AgentExecutor ready!")
    return agent


async def run_gsuite_agent(query: str) -> str:
    """Runs the GSuite agent with the given query and returns its output."""
    gsuite_agent_executor = await create_gsuite_agent_executor()

    try:
        response = await gsuite_agent_executor.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )

        final_output = "GSuite agent did not provide a recognizable output."
        if isinstance(response, dict):
            if (
                "messages" in response
                and isinstance(response["messages"], list)
                and response["messages"]
            ):
                last_message = response["messages"][-1]
                if (
                    hasattr(last_message, "content")
                    and not (
                        hasattr(last_message, "tool_calls") and last_message.tool_calls
                    )
                    and not hasattr(last_message, "type")
                    or last_message.type != "tool"
                ):
                    final_output = last_message.content

        if not isinstance(final_output, str):
            final_output = str(final_output)  # Ensure it's a string
        return final_output
    except Exception as e:
        raise RuntimeError(f"Error running GSuite agent: {str(e)}")


gsuite_tool = Tool(
    name="GSuiteAssistant",
    func=None,
    coroutine=run_gsuite_agent,
    description="Use this tool for any questions or tasks specifically related to accessing or managing GSuite services like Gmail and Google Calendar. Input should be the original user query about email or calendar.",
)


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


async def create_central_llm_with_tools():
    """Creates the Central LLM Agent."""
    llm = create_llm()

    tools = [gsuite_tool]  # Only GSuite tool for now

    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools


CENTRAL_LLM_TOOLS = [gsuite_tool]
