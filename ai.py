import asyncio
import logging
import os

import streamlit as st
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

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)

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

    return agent


async def process_google_mcp_response(agent, user_input: str, chat_history: list):
    """
    Processes the user input using the Google MCP agent.
    Returns the agent's response string if a tool was used and a response generated,
    otherwise returns None.
    """
    try:
        logging.info(
            f"MCP Agent processing input: '{user_input}' with history length: {len(chat_history)}"
        )
        response = await agent.run(user_input)
        logging.info(f"MCP Agent raw response: '{response}'")

        if response and response.strip():
            # If the agent provides a non-empty response,
            # we assume it handled the input.
            logging.info("MCP Agent provided a substantive response.")
            return response
        else:
            # If agent returns None, empty string, or only whitespace,
            # assume it did not handle the input with a tool.
            logging.info(
                "MCP Agent did not provide a substantive response, will fall back to main chain."
            )
            return None
    except Exception as e:
        logging.error(f"Error during Google MCP agent processing: {str(e)}")
        return None  # Fallback: if agent fails, return None so main chain can be tried.


def try_get_mcp_response(agent, user_input, chat_history):
    ai_response_content = None
    if agent:
        st.sidebar.info("⚙️ Checking MCP agent for response...")
        try:
            mcp_response = asyncio.run(
                process_google_mcp_response(
                    agent,
                    user_input,
                    chat_history,  # Pass full history for context
                )
            )
            if mcp_response:
                ai_response_content = mcp_response
                st.sidebar.success("✅ MCP agent handled the query.")
            else:
                st.sidebar.info(
                    "ℹ️ MCP agent did not handle the query, using standard chain."
                )
        except Exception as e:
            st.error(f"Error processing MCP response: {str(e)}")
            st.sidebar.error("⚠️ Error with MCP agent, falling back to standard chain.")
    else:
        st.sidebar.info(
            "ℹ️ MCP agent not available or not initialized, using standard chain."
        )
    return ai_response_content
