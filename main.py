import asyncio

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from ai import (
    create_chain,
    create_google_mcp_agent,
    create_llm,
    process_google_mcp_response,
)


def main():
    load_dotenv()

    st.title(
        "Siri but now when I ask it questions it will actually give me the answer and not just show me responses from google"
    )
    st.caption("A personal assistant to help with everyday tasks")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="How can I help you today?")]

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

    llm = create_llm()
    if llm is None:
        return

    chain = create_chain(llm)

    agent = None
    if llm:
        try:
            agent = create_google_mcp_agent(llm)
        except Exception as e:
            st.sidebar.error(
                f"❌ Exception during MCP agent creation in main.py: {str(e)}"
            )
            agent = None

    if user_input := st.chat_input("Type your message..."):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.write(user_input)

        ai_response_content = None

        if agent:
            st.sidebar.info("⚙️ Checking MCP agent for response...")
            try:
                mcp_response = asyncio.run(
                    process_google_mcp_response(
                        agent,
                        user_input,
                        st.session_state.chat_history,  # Pass full history for context
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
                st.sidebar.error(
                    "⚠️ Error with MCP agent, falling back to standard chain."
                )
        else:
            st.sidebar.info(
                "ℹ️ MCP agent not available or not initialized, using standard chain."
            )

        if ai_response_content:
            with st.chat_message("assistant"):
                st.write(ai_response_content)
            st.session_state.chat_history.append(AIMessage(content=ai_response_content))
        else:
            # Fallback to standard chain if MCP did not provide a response
            # or agent not available
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response_streamed = ""
                for chunk in chain.stream(
                    {
                        "chat_history": st.session_state.chat_history,
                        "input": user_input,
                    }
                ):
                    full_response_streamed += chunk.content
                    response_container.markdown(full_response_streamed + "▌")

                response_container.markdown(full_response_streamed)
                if full_response_streamed:  # Add to history only if there's content
                    st.session_state.chat_history.append(
                        AIMessage(content=full_response_streamed)
                    )


if __name__ == "__main__":
    main()
