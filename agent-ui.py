import streamlit as st
import importlib
import os

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

# Function to get available agent modules and their names
def get_available_agents():
    agents = {}
    for file in os.listdir('.'):
        if file.endswith('_agent.py'):
            module_name = file[:-3]  # Remove .py
            try:
                module = importlib.import_module(module_name)
                temp_agent = module.Agent()
                agents[module_name] = temp_agent.name
            except Exception as e:
                print(f"Error loading {module_name}: {str(e)}")
    return agents

# Add agent selector to sidebar
available_agents = get_available_agents()
selected_agent = st.sidebar.selectbox(
    "Select Agent Type",
    options=list(available_agents.keys()),
    format_func=lambda x: available_agents[x],
    key="agent_selector"
)

# Dynamic import of selected agent
if "current_agent_type" not in st.session_state:
    st.session_state.current_agent_type = selected_agent

# If agent type changed, reset the session
if st.session_state.current_agent_type != selected_agent:
    st.session_state.current_agent_type = selected_agent
    if "agent" in st.session_state:
        del st.session_state.agent
    if "messages" in st.session_state:
        st.session_state.messages = []

# Initialize agent
if "agent" not in st.session_state:
    try:
        module = importlib.import_module(selected_agent)
        st.session_state.agent = module.Agent()
    except Exception as e:
        st.error(f"Error loading agent: {str(e)}")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display title with agent name
st.title(f"Chat with {st.session_state.agent.name}")

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
