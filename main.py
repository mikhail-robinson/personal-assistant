import asyncio

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from ai import (
    create_tool_enhanced_agent,
)

load_dotenv()


async def main():
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

    agent = await create_tool_enhanced_agent()

    if user_input := st.chat_input("Type your message..."):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response_streamed = ""
            async for chunk in agent.astream(
                {
                    "chat_history": st.session_state.chat_history,
                    "input": user_input,
                },
            ):
                # Handle final output
                if "output" in chunk:
                    full_response_streamed += chunk["output"]
                    response_container.markdown(full_response_streamed + "▌")

                # Optional: handle tool actions or messages if you want to show them
                elif "actions" in chunk:
                    print("Tool action:", chunk["actions"])

        response_container.markdown(full_response_streamed)
        if full_response_streamed:  # Add to history only if there's content
            st.session_state.chat_history.append(
                AIMessage(content=full_response_streamed)
            )


if __name__ == "__main__":
    asyncio.run(main())
