import os
import openai
import fitz  
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

import streamlit as st
from tempfile import NamedTemporaryFile

# Load the API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def load_files(uploaded_files):
    temp_files = []
    for uploaded_file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_files.append(temp_file.name)
    return temp_files

st.title("Conversational Retrieval Chain with LangChain")
st.write("Upload your PDF files and ask questions based on their content.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    temp_file_paths = load_files(uploaded_files)
    text_data = []

    for file_path in temp_file_paths:
        text = extract_text_from_pdf(file_path)
        with NamedTemporaryFile(delete=False, suffix=".txt") as temp_text_file:
            temp_text_file.write(text.encode())
            text_data.append(temp_text_file.name)

    loaders = [TextLoader(file_path) for file_path in text_data]

    try:
        if PERSIST and os.path.exists("persist"):
            st.write("Reusing index...\n")
            vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            if PERSIST:
                index = VectorstoreIndexCreator(
                    vectorstore_kwargs={"persist_directory": "persist"},
                    embedding=OpenAIEmbeddings()
                ).from_loaders(loaders)
            else:
                index = VectorstoreIndexCreator(
                    embedding=OpenAIEmbeddings()
                ).from_loaders(loaders)

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        )

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Enter your question:")

        if query:
            result = chain({"question": query, "chat_history": st.session_state.chat_history})
            st.write(result['answer'])

            st.session_state.chat_history.append((query, result['answer']))

        if st.button("Clear History"):
            st.session_state.chat_history = []

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.write("Please upload PDF files to proceed.")
