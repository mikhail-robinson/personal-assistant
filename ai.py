import datetime
import os

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
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
        f"IMPORTANT: The current date and time is {today}. You are a general-purpose AI assistant. Your FIRST priority is to answer questions using your internal knowledge. "
        "For example, if asked 'What is a bee?', you MUST provide a factual answer from your knowledge base. Do NOT say your capabilities are limited. "
        "You ALSO have specialized tools for interacting with services like Gmail and Google Calendar. "
        "You should ONLY use these tools if a user's query EXPLICITLY asks for information from, or an action on, these specific services (e.g., 'read my emails', 'check my calendar', 'create an event'). "
        "If a query is general (e.g., 'What is the capital of France?', 'Tell me about photosynthesis'), answer it directly using your knowledge. "
        "If you use a tool and it finds no information (e.g., no unread emails), clearly state that (e.g., 'I checked your Gmail, and there are no unread emails.'). "
        "Do not respond with generic phrases like 'I don't have a specific response' if you can answer from knowledge or if a tool result can be clearly explained."
    )
    print(today)

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
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


async def create_tool_enhanced_agent():
    llm = create_llm()
    prompt = create_agent_prompt()
    tools = await google_mcp_tools()
    if not tools:
        return None

    llm_with_tools = llm.bind_tools(tools)

    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)  # Create the agent

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    st.sidebar.success("✅ Tool-enhanced AgentExecutor ready!")
    return agent_executor
