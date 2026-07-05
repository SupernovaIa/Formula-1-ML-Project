# Environment configuration
# -----------------------------------------------------------------------
import os
import dotenv

# LangChain utilities for document processing and retrieval
# -----------------------------------------------------------------------
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


dotenv.load_dotenv()

DOCS_DIR = "docs/"

SYSTEM_PROMPT = """You are an expert assistant for Formula 1 fans.

Answer only using the retrieved context below. If the context doesn't cover
the question, say so directly instead of guessing. Keep answers direct,
informative and concise.

IMPORTANT: always reply in the same language as the user's question, even if
the retrieved context below is in a different language. Translate any facts
you use from the context — never answer in the context's language by default.
This applies even when you have no answer: the "I don't have information on
that" message itself must still be in the user's question's language.

{context}"""


def load_documents(docs_dir=DOCS_DIR):
    """
    Loads every supported document (.txt, .md) from a directory.

    Parameters
    -----------
    - docs_dir (str): Directory to load documents from.

    Returns
    --------
    - (list): Loaded LangChain document objects.
    """
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Directory '{docs_dir}' does not exist.")

    supported_types = {"txt", "md"}
    docs = []

    for filename in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, filename)
        if os.path.isfile(file_path) and filename.split(".")[-1] in supported_types:
            docs.extend(TextLoader(file_path).load())

    return docs


def build_vector_db(docs_dir=DOCS_DIR):
    """
    Builds an in-memory vector database from every document in `docs_dir`.

    Meant to be called once (e.g. at backend startup) and reused across
    requests — the corpus is the same for every user, so there's no need to
    rebuild it per session.

    Parameters
    -----------
    - docs_dir (str): Directory to load documents from.

    Returns
    --------
    - (Chroma or None): The vector database, or None if no documents were found.
    """
    docs = load_documents(docs_dir)

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(docs)

    return Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())


def _get_context_retriever_chain(vector_db, llm):
    """
    Creates a context-aware retriever chain for retrieving relevant information from a vector database.

    Parameters
    -----------
    - vector_db (Chroma): A Chroma vector database instance used for retrieving documents.
    - llm (object): A language model instance used to generate search queries.

    Returns
    --------
    - retriever_chain (object): A history-aware retriever chain for contextual information retrieval.
    """
    retriever = vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])

    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(vector_db, llm):
    """
    Creates a Conversational RAG (Retrieval-Augmented Generation) chain.

    Parameters
    -----------
    - vector_db (Chroma): A Chroma vector database instance used for retrieving documents.
    - llm (object): A language model instance used for generating responses.

    Returns
    --------
    - rag_chain (object): A retrieval chain that retrieves relevant context and generates responses.
    """
    retriever_chain = _get_context_retriever_chain(vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_rag_response(vector_db, llm_stream, messages):
    """
    Streams a RAG response for the given conversation.

    Parameters
    -----------
    - vector_db (Chroma): A Chroma vector database instance used for retrieving documents.
    - llm_stream (object): A streaming-capable chat model.
    - messages (list): Full conversation history (LangChain message objects);
      the last message is treated as the current turn's input.

    Yields
    -------
    - chunk (str): A streamed chunk of the response text.
    """
    conversation_rag_chain = get_conversational_rag_chain(vector_db, llm_stream)

    yield from conversation_rag_chain.pick("answer").stream({
        "messages": messages[:-1],
        "input": messages[-1].content,
    })
