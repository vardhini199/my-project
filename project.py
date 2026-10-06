
import os
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="University AI Chatbot",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FOLDER = "documents"
CHROMA_FOLDER = "chroma_db"

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎓 University AI Information Chatbot")

st.write(
    "Ask questions about university courses, admissions, "
    "departments, fees, facilities, examinations and other information."
)


# ============================================================
# LOAD TEXT DOCUMENT
# ============================================================

def load_documents():

    documents = []

    # Create documents folder if it does not exist
    if not os.path.exists(DOCUMENT_FOLDER):
        os.makedirs(DOCUMENT_FOLDER)

    # Read all TXT files
    for file_name in os.listdir(DOCUMENT_FOLDER):

        if file_name.lower().endswith(".txt"):

            file_path = os.path.join(
                DOCUMENT_FOLDER,
                file_name
            )

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    text = file.read()

                if text.strip():

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_name
                            }
                        )
                    )

            except Exception as e:

                st.error(
                    f"Error reading {file_name}: {e}"
                )

    return documents


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

@st.cache_resource
def create_vector_database():

    documents = load_documents()

    if not documents:
        return None

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # Create embeddings
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Create Chroma vector database
    vector_database = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER
    )

    return vector_database


# ============================================================
# CREATE OLLAMA AI MODEL
# ============================================================

@st.cache_resource
def create_llm():

    return OllamaLLM(
        model=LLM_MODEL,
        temperature=0.2
    )


# ============================================================
# LOAD AI SYSTEM
# ============================================================

with st.spinner("Loading university information..."):

    try:

        vector_database = create_vector_database()

        llm = create_llm()

    except Exception as e:

        st.error("Unable to start the AI system.")

        st.error(str(e))

        st.stop()


# ============================================================
# CHECK DOCUMENTS
# ============================================================

if vector_database is None:

    st.warning(
        "⚠️ No university documents found."
    )

    st.info(
        "Please place university_information.txt "
        "inside the documents folder."
    )

    st.stop()


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = vector_database.as_retriever(
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# QUESTION ANSWERING FUNCTION
# ============================================================

def ask_question(question):

    # Search relevant information
    documents = retriever.invoke(
        question
    )

    # If no information is found
    if not documents:

        return (
            "I could not find this information "
            "in the university documents.",
            []
        )

    # Combine retrieved text
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # AI prompt
    prompt = f"""
You are an AI University Information Assistant.

Your job is to answer questions about the university.

IMPORTANT RULES:

1. Use ONLY the information provided in the context.
2. Do not invent or assume information.
3. Give clear and simple answers.
4. If the answer is not available in the context,
   say exactly:

"I could not find this information in the university documents."

5. Do not provide information that is not supported
   by the university documents.

----------------------------------------
UNIVERSITY INFORMATION
----------------------------------------

{context}

----------------------------------------
USER QUESTION
----------------------------------------

{question}

----------------------------------------
ANSWER
----------------------------------------

"""

    # Generate answer
    answer = llm.invoke(
        prompt
    )

    return answer, documents


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

user_question = st.chat_input(
    "Ask your university question..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_question:

    # Display user question
    with st.chat_message("user"):

        st.markdown(
            user_question
        )

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching university information..."
        ):

            try:

                answer, source_documents = ask_question(
                    user_question
                )

                st.markdown(
                    answer
                )

                # Show sources
                if source_documents:

                    with st.expander(
                        "📚 View Sources"
                    ):

                        shown_sources = set()

                        for document in source_documents:

                            source = document.metadata.get(
                                "source",
                                "University document"
                            )

                            if source not in shown_sources:

                                st.write(
                                    f"📄 {source}"
                                )

                                shown_sources.add(
                                    source
                                )

            except Exception as e:

                answer = (
                    "Sorry, an error occurred "
                    "while processing your question."
                )

                st.error(
                    str(e)
                )

    # Store AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 University Chatbot")

    st.write(
        "AI-powered University Information "
        "Retrieval System"
    )

    st.divider()

    st.subheader("💡 Example Questions")

    st.write(
        "• What courses are offered?"
    )

    st.write(
        "• What departments are available?"
    )

    st.write(
        "• How can I apply for admission?"
    )

    st.write(
        "• What is the eligibility criteria?"
    )

    st.write(
        "• What are the hostel facilities?"
    )

    st.write(
        "• What scholarships are available?"
    )

    st.write(
        "• What examination system is followed?"
    )

    st.write(
        "• Does the university provide placement assistance?"
    )

    st.divider()

    # Clear chat button
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

