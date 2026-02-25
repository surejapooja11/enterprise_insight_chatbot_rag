✅ Step 1 — Clean Project Structure
Inside:
enterprise_insight_chatbot_rag
Create a new folder called:
data
Move your 3 PDFs into that folder.
So it becomes:
enterprise_insight_chatbot_rag/
│
├── data/
│     ├── apple10k-2024.pdf
│     ├── ibm10k-2024.pdf
│     └── intel10k-2024.pdf
│
├── cleanrag #environment - all library installed in this environment
├── requirements.txt  - it has all library written which we need to install for rag app
├── app.py -rag app run code
├── readme.txt
├──.env - has api key


✅ Step 2 — Open Terminal in Project Folder
On Mac:
Open Terminal and run:
cd ~/Desktop/ai-projects/enterprise_insight_chatbot_rag


✅ Step 3 — Install Dependencies

-- fully exit enaconda conda deactivate - close terminal and rerun

-- Create a new venv inside your project folder
cd ~/Desktop/ai_projects/enterprise_insight_chatbot_rag
python3 -m venv cleanrag

-- Activate new environment
source cleanrag/bin/activate

pip install --upgrade pip

-- intall packages 
pip install -r requirements.txt

1️⃣ langchain
The core framework for building RAG pipelines and connecting LLMs to data sources.
Handles document chains, prompts, and retrieval logic.
2️⃣ langchain-community
Community-contributed extensions for LangChain.
Includes PyPDFLoader and other document loaders not in the core package.
3️⃣ langchain-openai
Integrates OpenAI LLMs (like GPT-4o-mini) with LangChain.
Lets your chatbot query documents and generate answers.
4️⃣ openai
Official OpenAI Python SDK.
Needed for API calls to models like GPT-4, embeddings, and chat completions.
5️⃣ faiss-cpu
Vector database for embeddings.
Stores and searches document vectors efficiently for retrieval.
6️⃣ pypdf
Reads and parses PDF files.
Required by PyPDFLoader to convert PDFs into text chunks.
7️⃣ pdfminer.six
Alternative PDF parser.
Some PDFs require this to extract text correctly.
8️⃣ tiktoken
OpenAI’s tokenizer library.
Required for calculating token counts in documents and LLM calls.
9️⃣ python-dotenv
Loads your API keys and environment variables from a .env file.
Keeps your keys secure and separate from code.

✅ Step 4 — Run app

python app.py # inside your environment