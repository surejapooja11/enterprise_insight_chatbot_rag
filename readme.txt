Enterprise Insight Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers financial questions 
based on company 10-K PDFs using OpenAI GPT-4o-mini and LangChain.

------------------------------------------------------------
✅ Step 1 — Project Structure

Organize your project like this:

enterprise_insight_chatbot_rag/
├── data/                  # Folder for 10-K PDFs
│     ├── apple10k-2024.pdf
│     ├── ibm10k-2024.pdf
│     └── intel10k-2024.pdf
├── cleanrag/              # Python virtual environment (do NOT push to GitHub)
├── requirements.txt       # Python dependencies
├── app.py                 # Main RAG chatbot script
├── README.txt             # Project description (this file)
├── .env.example           # Example API key file

Note: Keep your .env file private and do not upload it to GitHub. Provide .env.example instead.

------------------------------------------------------------
✅ Step 2 — Open Terminal in Project Folder

Navigate to your project folder in the terminal:

cd ~/Desktop/ai_projects/enterprise_insight_chatbot_rag

------------------------------------------------------------
✅ Step 3 — Install Dependencies

1. Exit any existing Anaconda environments:

conda deactivate

2. Create a new virtual environment:

python3 -m venv cleanrag

3. Activate the environment:

source cleanrag/bin/activate   # Mac/Linux
# cleanrag\Scripts\activate    # Windows

4. Upgrade pip and install packages:

pip install --upgrade pip
pip install -r requirements.txt

------------------------------------------------------------
Dependencies Explained

- langchain: Core framework for building RAG pipelines; handles document chains, prompts, and retrieval logic.
- langchain-community: Community-contributed extensions for LangChain, including PyPDFLoader.
- langchain-openai: Integrates OpenAI LLMs (like GPT-4o-mini) with LangChain.
- openai: Official OpenAI Python SDK for API calls, embeddings, and chat completions.
- faiss-cpu: Vector database for embeddings; efficient storage and search of document vectors.
- pypdf: Reads and parses PDF files.
- pdfminer.six: Alternative PDF parser for PDFs that require extra parsing.
- tiktoken: OpenAI tokenizer library; calculates token counts for LLM calls.
- python-dotenv: Loads environment variables from .env file; keeps API keys secure.

------------------------------------------------------------
✅ Step 4 — Run the Chatbot

After installing dependencies and activating your environment, run:

python app.py

- Type your questions in the terminal.
- Type 'exit' to quit the chatbot.

Note: The chatbot only answers based on the provided PDFs; it does not use external knowledge.

------------------------------------------------------------
✅ Step 5 — Adding PDFs

Place your 10-K PDFs in the data/ folder with these names:

apple10k-2024.pdf
ibm10k-2024.pdf
intel10k-2024.pdf

You can also use sample PDFs for testing if the full PDFs are large.

------------------------------------------------------------
✅ Step 6 — Notes

- Keep your virtual environment (cleanrag/) and .env file out of GitHub.
- This project demonstrates skills in: Python, LangChain, OpenAI LLMs, FAISS vector databases, and PDF processing.
- anyone can clone the repo, add their own PDFs, and run the chatbot immediately.

------------------------------------------------------------