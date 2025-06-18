import asyncio

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from ai import create_central_llm_agent # Updated import

load_dotenv()


async def main():
    langfuse = get_client()
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

    agent = await create_central_llm_agent() # Updated agent creation

    if user_input := st.chat_input("Type your message..."):
        user_message = HumanMessage(content=user_input)
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with langfuse.start_as_current_span(name="langchain-request") as span:
                span.update_trace(input={"input": user_input})
                handler = CallbackHandler()
                response_container = st.empty()
                response_container.markdown("Hmmmm...")

                full_response_content = ""

                async for chunk in agent.astream(
                    {"messages": st.session_state.chat_history},
                    stream_mode="messages",
                    config={"callbacks": [handler]},
                ):
                    # chunk is a tuple: (AIMessageChunk, metadata_dict)
                    if len(chunk) >= 1:
                        message_chunk = chunk[0]  # Get the AIMessageChunk
                        if (
                            hasattr(message_chunk, "content")
                            and message_chunk.content
                            and not (
                                hasattr(message_chunk, "tool_calls")
                                and message_chunk.tool_calls
                            )
                            and not hasattr(message_chunk, "type")
                            or message_chunk.type != "tool"
                        ):
                            full_response_content += message_chunk.content
                            response_container.markdown(full_response_content)
                if full_response_content:
                    st.session_state.chat_history.append(
                        AIMessage(content=full_response_content)
                    )


if __name__ == "__main__":
    asyncio.run(main())
