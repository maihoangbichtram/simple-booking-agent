"""Streamlit UI for the clinic booking chatbot."""

import streamlit as st
from booking_agent import create_initial_state, process_message

def initialize_session():
    """Initialize session state."""
    if "state" not in st.session_state:
        st.session_state.state = create_initial_state()
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


def display_chat_history():
    """Display the chat history with persistent options and styling."""
    messages = st.session_state.state.get("messages", [])
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])

                # Show options if they exist
                options = message.get("options", [])
                if options:
                    # If this is the last message in history, show as clickable buttons
                    if i == len(messages) - 1 and st.session_state.state["stage"] not in ["completed", "cancelled"]:
                        st.markdown("---")
                        # Create columns for buttons
                        cols = st.columns(min(len(options), 3))
                        for idx, option in enumerate(options):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                if st.button(option, key=f"btn_{i}_{idx}", use_container_width=True):
                                    handle_user_input(option)
                    else:
                        # For older messages, show options as pills/text to keep history
                        options_str = "  ".join([f"`{opt}`" for opt in options])
                        st.markdown(f"**Available options:** {options_str}")
        else:
            with st.chat_message("user"):
                st.markdown(message["content"])


def handle_user_input(user_input: str):
    """Handle user input and process through agent."""
    # Process the message
    st.session_state.state = process_message(
        st.session_state.state,
        user_input,
        thread_id=st.session_state.session_id
    )

    # Rerun to update UI
    st.rerun()


def run_chat_ui():
    """Run the chat UI."""
    # Page config
    st.set_page_config(
        page_title="Book Appointment",
        page_icon="🏥",
        layout="centered"
    )

    # Custom CSS for distinction between messages
    st.markdown("""
        <style>
        [data-testid="stChatMessageUser"] {
            flex-direction: row-reverse;
            text-align: right;
            background-color: #e0f2f1;
            border-radius: 15px 15px 0px 15px;
        }
        [data-testid="stChatMessageAssistant"] {
            background-color: #f5f5f5;
            border-radius: 15px 15px 15px 0px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session
    initialize_session()

    # Header
    st.title("🏥 Booking Page")
    st.markdown("*Book your appointment easily*")
    st.markdown("---")

    # Send initial greeting if not initialized
    if not st.session_state.initialized:
        st.session_state.state = process_message(
            st.session_state.state,
            "Hi",
            thread_id=st.session_state.session_id
        )
        st.session_state.initialized = True
        st.rerun()

    # Display chat history
    display_chat_history()

    # Chat input (only show if not completed)
    if st.session_state.state["stage"] not in ["completed", "cancelled"]:
        if prompt := st.chat_input("Type your message here..."):
            handle_user_input(prompt)
    else:
        # Show restart button after completion
        st.markdown("---")
        if st.button("🔄 Start New Booking", use_container_width=True):
            st.session_state.state = create_initial_state()
            st.session_state.initialized = False
            st.rerun()