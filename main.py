import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from ai import create_chain, create_llm


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

    if user_input := st.chat_input("Type your message..."):
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Process AI response only if there's new user input
        response_container = st.empty()
        full_response = ""

        # Stream the response from the chain
        for chunk in chain.stream(
            {
                "chat_history": st.session_state.chat_history,  # Pass the updated history
                "input": user_input,  # Pass the new user input
            }
        ):
            full_response += chunk.content
            response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)
        st.session_state.chat_history.append(AIMessage(content=full_response))


if __name__ == "__main__":
    main()
