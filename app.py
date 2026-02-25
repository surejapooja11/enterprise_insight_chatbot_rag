import os
import glob
from dotenv import load_dotenv #to load .env file so api key available
from langchain_community.document_loaders import PyPDFLoader  #this load pdf and convert each page in langchain document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI #Openaiembedding - convert text into numerical vectors and Chatopenai - the llm that generate the final answers
# from langchain.chains.retrieval_qa.base import RetrievalQA #connects everything together - retriver + llm -> question answering system
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

#Load Api Key
load_dotenv()

# --- Load PDFs dynamically from data folder ---
documents = []
pdf_files = glob.glob(os.path.join("data", "*.pdf"))  # automatically finds all PDFs

if not pdf_files:
    print("No PDF files found in 'data/' folder!")
else:
    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load()

        # Optional: add company metadata based on filename
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

print(f"Loaded {len(documents)} pages")


#Split Text - it broken into a smaller pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks")

#Create embedding - model convert text into vectors
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

#Create Vector Store - create embedding and each store in FAISS
vectorstore = FAISS.from_documents(texts,embeddings)

#Create LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

#Create Retrival Chain - core RAG (This connects: User Question -> Retriever → Finds relevant chunks -> LLM → Uses chunks + question-> Answers)
prompt = ChatPromptTemplate.from_template("""
You are a financial analyst answering questions strictly using the provided context.

Rules:
- Only use information explicitly found in the context.
- If information for ANY company mentioned in the question is missing from the context, say:
  "Insufficient information in provided documents."
- Do NOT use outside knowledge.
- Do NOT use prior knowledge.
- Do NOT make assumptions.

Context:
{context}

Question:
{input}

Answer:
""")

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 12}
)

# Create retrieval chain
qa = create_retrieval_chain(retriever, document_chain)

#Infinite Question Loop
while True:
    query = input("\nAsk a question (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    response = qa.invoke({"input": query})
    print("\nAnswer:\n", response["answer"])

