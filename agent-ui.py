import streamlit as st
from openai_agent import Agent  # Assuming Agent class exists

# Fix annoying UI issues
st.markdown(
    """
    <style>
    .stAppDeployButton {
        visibility: hidden;
    }
    .stSidebar {
            min-width: 200px;
            max-width: 200px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize agent
if "agent" not in st.session_state:
    st.session_state.agent = Agent()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Agent Chat")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from agent
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""

        try:
            response = st.session_state.agent.chat(user_input)
            response_text = str(response)
        except Exception as e:
            response_text = f"Error: {e}"

        response_container.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.agent.clear_chat()
    st.session_state.messages = []
    st.rerun()
