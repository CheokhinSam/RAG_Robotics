import streamlit as st
import models
from pathlib import Path
import os
from datetime import datetime
import re

if "memory" not in st.session_state:
    st.session_state.memory = models.ConversationBufferMemory()
if "db" not in st.session_state:
    st.session_state.db = None
if "collection_priority" not in st.session_state:
    st.session_state.collection_priority = False
if "current_collection" not in st.session_state:
    st.session_state.current_collection = "None"  

st.set_page_config(page_title="RAG Q&A System", layout="wide")
st.title("RAG project")

st.subheader("Select a Collection to Query")
try:
    collections = models.get_available_collections()
    if collections:
        selected_collection = st.selectbox("Choose a collection", collections, index=0, key="collection_select")
        if st.button("Load Selected Collection"):
            st.session_state.db = models.load_existing_vector_store(collection_name=selected_collection)
            st.session_state.collection_priority = True
            st.session_state.current_collection = selected_collection  
            st.success(f"Loaded collection: {selected_collection}")
    else:
        st.warning("No collections found in the database.")
except Exception as e:
    st.error(f"Failed to load collections: {str(e)}")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Select a collection or upload files.
    2. Ask questions related to the document.
    3. The system will retrieve content and provide a concise answer.
    Note: Selecting a collection takes priority over uploading files.
    """)
    st.header("Settings")
    st.markdown("""
    - LLM: azure-gpt-4o
    - Embedding Model: azure-text-embedding-3-large
    - Retriever Type: Similarity Search
    """)

    st.header("Current Collection")
    st.write(f"**{st.session_state.current_collection}**")

st.header("Upload Files (Optional)")
uploaded_files = st.file_uploader(
    "Upload your files here",
    type=["pdf", "docx", "txt", "png", "jpg", "csv", "md"],
    accept_multiple_files=True,
    help="You can upload multiple files at once"
)

if uploaded_files and not st.session_state.collection_priority:
    try:
        with st.spinner("Files uploaded, please wait for processing..."):
            file_paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            first_file_name = re.sub(r'[<>:"/\\|?*]', '', uploaded_files[0].name)
            base_name = os.path.splitext(first_file_name)[0]
            collection_name = f"{base_name}_{timestamp}"
            
            for uploaded_file in uploaded_files:
                safe_name = re.sub(r'[<>:"/\\|?*]', '', uploaded_file.name)
                file_path = Path(models.WORKING_DIR) / f"{timestamp}_{safe_name}"
                models.upload_file(uploaded_file, file_path)
                file_paths.append(str(file_path))

            st.session_state.db = models.create_vector_store(file_paths, collection_name=collection_name)
            st.session_state.current_collection = collection_name 
        st.success(f"{len(uploaded_files)} file(s) processed successfully! Collection: {collection_name}")

    except Exception as e:
        st.error(f"An error occurred while processing the files: {str(e)}")
        st.info("Please check the file formats or try again later.")
elif uploaded_files and st.session_state.collection_priority:
    st.info("Upload ignored. Using selected collection data.")

st.markdown("<h2 style='text-align: center;'>Teacher Assistant Chatbot</h2>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.db is None:
    st.info("No vector store available. Please select a collection or upload files.")
else:
    if question := st.chat_input("Ask a question here!"):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Generating answer..."):
            related_docs = models.retrieve_docs(st.session_state.db, question)
            answer = models.question_file(question, related_docs, st.session_state.memory)
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})