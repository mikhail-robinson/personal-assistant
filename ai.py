import logging
import os

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_use import MCPAgent, MCPClient


def create_llm():
    """Create a language model instance"""

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY not found in environment variables. Please add it to your .env file."
        )
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)

    return llm


def create_chain(llm):
    """Create a language chain with conversation memory"""

    prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
    )

    chain = prompt | llm
    return chain


def create_google_mcp_agent(llm):
    """Create an MCP agent for accessing the google suite"""

    logger = logging.getLogger("chatbot")
    logger.info("Initializing MCP agent")

    config = {
        "mcpServers": {
            "mcp-gsuite": {
                "command": "uv",
                "args": [
                    "--directory",
                    "/Users/mikhail/Documents/AI Projects/mcp-gsuite",
                    "run",
                    "mcp-gsuite",
                ],
            }
        }
    }

    try:
        # Log the actual configuration
        logger.info(f"MCP config: {config}")

        # Create the client
        client = MCPClient.from_dict(config)

        # Add status message to UI
        st.sidebar.success("✅ MCP Docker configuration initialized")

    except Exception as e:
        logger.error(f"Failed to create MCP client: {str(e)}")
        st.sidebar.error(f"❌ Failed to configure MCP: {str(e)}")
        # Create a fallback client
        client = MCPClient.from_dict(config)

    logger.info("Creating MCP agent")

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=30,
        verbose=True,
        # system_message parameter intentionally omitted for Gemini compatibility
    )

    logger.info("MCP agent successfully created")
    st.sidebar.success("✅ MCP agent ready!")

    return agent, client


async def process_google_mcp_response(agent, client, user_input):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        status_placeholder.markdown("🔄 Running agent...")

        st.sidebar.info("🌐 Agent is accessing Google MCP...")
        full_response = await agent.run(user_input)

        st.sidebar.success("✅ Agent has succesfully accessed the Google MCP!")

        message_placeholder.markdown(full_response)
        status_placeholder.empty()
        st.session_state.chat_history.append(AIMessage(content=full_response))

        return full_response
