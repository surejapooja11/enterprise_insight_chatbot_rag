import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import glob
import os

# Load API Key
load_dotenv()

st.title("📊 Enterprise RAG Chatbot")
st.write("Ask questions about company 10-K PDFs. The answers come strictly from the provided documents.")

# --- Load PDFs ---
@st.cache_resource
# --- Load all PDFs in the data folder ---
def load_documents():
    documents = []
    pdf_files = glob.glob(os.path.join("data", "*.pdf"))  # finds all .pdf files

    if not pdf_files:
        st.error("No PDF files found in the 'data' folder!")
        return documents

    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load()
        
        # Add company metadata from filename (optional)
        filename = os.path.basename(file).lower()
        if "apple" in filename:
            company = "Apple"
        elif "ibm" in filename:
            company = "IBM"
        elif "intel" in filename:
            company = "Intel"
        else:
            company = "Unknown"

        for doc in docs:
            doc.metadata["company"] = company

        documents.extend(docs)

    return documents

documents = load_documents()

# --- Split Text ---
@st.cache_resource
def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

vectorstore = create_vectorstore(documents)

# --- LLM & QA chain ---
@st.cache_resource
def create_qa_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
You are a financial analyst answering questions strictly using the provided context.

Rules:
- Only use information explicitly found in the context.
- If information for ANY company mentioned in the question is missing, say:
  "Insufficient information in provided documents."
- Do NOT use outside knowledge.

Context:
{context}

Question:
{input}

Answer:
""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    qa = create_retrieval_chain(retriever, document_chain)
    return qa

qa = create_qa_chain()

# --- Chat input/output ---
query = st.text_input("Type your question here:")

if query:
    with st.spinner("Fetching answer..."):
        response = qa.invoke({"input": query})
    st.text_area("Answer:", value=response["answer"], height=200)