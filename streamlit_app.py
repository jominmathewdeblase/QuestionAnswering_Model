import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Layout for API keys input
api_keys = st.columns(2)
groq_api_key_col, google_api_key_col = api_keys

groq_api_key = groq_api_key_col.text_input("GROQ API Key", type="password")
google_api_key = google_api_key_col.text_input("Google API Key", type="password")

# Check if both keys are entered
if not groq_api_key or not google_api_key:
    st.error("Please enter both GROQ and Google API keys to continue.", icon="❗️")
else:
    st.title("Deblase Document Q&A")

    llm = ChatGroq(groq_api_key=groq_api_key,
                   model_name="llama3-groq-70b-8192-tool-use-preview")

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    def vector_embedding():
        # Check if vectors already exist in session state
        if "vectors" not in st.session_state:
            try:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.loader = PyPDFDirectoryLoader("./debase_doc_c")  # Data Ingestion
                st.session_state.docs = st.session_state.loader.load()  # Document Loading

                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting

                st.write("Loaded documents:", len(st.session_state.final_documents))  # Debug statement
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

            except Exception as e:
                st.error(f"Error during vector embedding: {str(e)}")  # Display any errors

    # User input for question
    prompt1 = st.text_input("Enter Your Question From Documents")

    # Button to trigger embedding
    if st.button("Documents Embedding"):
        vector_embedding()
        st.write("Vector Store DB Is Ready")

    # Check if vectors exist before processing the question
    if prompt1:
        if "vectors" not in st.session_state:
            st.warning("Please click 'Documents Embedding' first to initialize the vector store.")
        else:
            try:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start

                st.write("Response time:", response_time)

                if 'answer' in response:
                    st.write(response['answer'])
                else:
                    st.write("No answer found in the response.")

                # Display relevant context
                with st.expander("Document Similarity Search"):
                    if "context" in response:
                        for i, doc in enumerate(response["context"]):
                            st.write(doc.page_content)
                            st.write("--------------------------------")
                    else:
                        st.write("No context found.")

            except Exception as e:
                st.error(f"Error during retrieval: {str(e)}")  # Handle retrieval errors
