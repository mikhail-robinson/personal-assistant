import asyncio

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from ai import (
    CENTRAL_LLM_TOOLS,
    create_central_llm_prompt,
    create_central_llm_with_tools,
)  # Updated import

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

    llm_with_tools = await create_central_llm_with_tools()
    prompt = create_central_llm_prompt()

    chain = prompt | llm_with_tools

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

                full_initial_stream_content = ""
                accumulated_initial_message_chunk = None

                # Stream initial message response
                async for chunk in chain.astream(
                    {"messages": st.session_state.chat_history},
                    stream_mode="messages",
                    config={"callbacks": [handler]},
                ):
                    # output the streamed response to the user
                    if chunk.content:
                        full_initial_stream_content += chunk.content
                        response_container.markdown(full_initial_stream_content)

                    # get the whole chunk to check for tool calls
                    if accumulated_initial_message_chunk is None:
                        accumulated_initial_message_chunk = chunk
                    else:
                        accumulated_initial_message_chunk += chunk
                if accumulated_initial_message_chunk:
                    st.session_state.chat_history.append(
                        accumulated_initial_message_chunk
                    )

                full_response_content = ""

                if (
                    accumulated_initial_message_chunk
                    and accumulated_initial_message_chunk.tool_calls
                ):
                    tool_messages_for_llm = []
                    for tool_call in accumulated_initial_message_chunk.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        tool_function = None
                        for tool in CENTRAL_LLM_TOOLS:
                            if tool.name == tool_name:
                                tool_function = tool
                                break

                        if tool_function:
                            tool_result = await tool_function.ainvoke(tool_args)
                            tool_message = ToolMessage(
                                content=str(tool_result), tool_call_id=tool_call["id"]
                            )
                            tool_messages_for_llm.append(tool_message)

                            st.session_state.chat_history.extend(tool_messages_for_llm)

                    async for chunk in chain.astream(
                        {"messages": st.session_state.chat_history},
                        stream_mode="messages",
                        config={"callbacks": [handler]},
                    ):
                        full_response_content += chunk.content
                        response_container.markdown(full_response_content)
                    response_container.markdown(full_response_content)
                    if full_response_content:
                        st.session_state.chat_history.append(
                            AIMessage(content=full_response_content)
                        )


if __name__ == "__main__":
    asyncio.run(main())
