import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from src.rag import *

# Load dot env
dotenv.load_dotenv()

# List of AI models
MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o"
]

# Streamlit page config
st.set_page_config(
    page_title="F1 Chatbot 🏁", 
    page_icon="🏎️", 
    layout="centered", 
    initial_sidebar_state="expanded"
)


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]

# --- Side Bar LLM API Tokens ---
with st.sidebar:

    default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
    with st.popover("OpenAI API Key"):
        openai_api_key = st.text_input(
            "Introduce your OpenAI API Key (https://platform.openai.com/)", 
            value=default_openai_api_key, 
            type="password",
            key="openai_api_key",
        )


# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key

if missing_openai:
    st.warning("⬅️ Please introduce an API Key to continue!")

else:
    pass