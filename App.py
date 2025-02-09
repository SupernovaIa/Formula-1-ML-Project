# Web application framework
# -----------------------------------------------------------------------
import streamlit as st

# Environment and system utilities
# -----------------------------------------------------------------------
import os
import dotenv
import uuid

# check if it's linux so it works on Streamlit Cloud
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# LangChain models and schemas
# -----------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Custom modules
# -----------------------------------------------------------------------
from src.rag import load_doc_to_db, stream_llm_rag_response, stream_llm_response

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

# Initial setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]

# API Key
with st.sidebar:

    default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
    
    with st.popover("OpenAI API Key", icon="🔒"):
        openai_api_key = st.text_input(
            "Introduce your OpenAI API Key", 
            value=default_openai_api_key, 
            type="password",
            key="openai_api_key"
        )

# Checking if the user has introduced the OpenAI API Key
missing_openai = openai_api_key is None or "sk-" not in openai_api_key

if missing_openai:
    st.warning("⬅️ Please introduce an API Key to continue!")

else:
    # Sidebar
    with st.sidebar:
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model.split("/")[-1])

        st.selectbox(
            "Model", 
            options=models,
            key="model",
            help="Currently, only OpenAI models are supported"
        )

        # Load RAG documents
        is_vector_db_loaded = False
        st.button("Load RAG docs", on_click=load_doc_to_db, type="secondary", disabled = is_vector_db_loaded)

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded
            )

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        temp = st.slider("Temperatue", min_value=0.0, max_value=1.0, value=0.3, step=0.1, help="Select model temperature.")

    # Main chat app
    llm_stream = ChatOpenAI(
        api_key=openai_api_key,
        model_name=st.session_state.model.split("/")[-1],
        temperature=temp,
        streaming=True,
    )

    # Message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Prompt and chat
    if prompt := st.chat_input("Type here your message"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            if st.session_state.use_rag:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_response(llm_stream, messages))
