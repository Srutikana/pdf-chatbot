import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables
load_dotenv()

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Streamlit app title
st.title("📄 Chat with Multiple PDFs")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    process_button = st.button("Process PDFs")

# Function to extract text from PDFs
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
    return text

# Function to process text and create vector store
def process_pdfs(pdf_files):
    try:
        # Extract text
        raw_text = get_pdf_text(pdf_files)
        if not raw_text:
            st.error("No text extracted from PDFs. Ensure PDFs contain readable text.")
            return

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(raw_text)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Initialize language model with explicit task
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",  # Explicitly set the task
            max_new_tokens=512,
            temperature=0.7,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create conversational retrieval chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.conversation = conversation_chain
        st.success("PDFs processed successfully!")
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")

# Process uploaded PDFs
if process_button and uploaded_files:
    with st.spinner("Processing PDFs..."):
        process_pdfs(uploaded_files)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about the PDFs"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response if conversation chain is initialized
    if st.session_state.conversation:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({"question": prompt})
                    answer = response["answer"]
                    st.write(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    else:
        with st.chat_message("assistant"):
            st.write("Please upload and process PDFs first.")