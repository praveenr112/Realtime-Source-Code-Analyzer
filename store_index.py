from src.helper import repo_ingestion,load_repo,text_splitter,load_embeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# url = "https://github.com/praveenr112/End-to-End-Medical-Chatbot"

# repo_ingestion(url)

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embeddings()

#Storing vector in chromadb

vectordb = Chroma.from_documents(text_chunks,embedding=embeddings,persist_directory='./db')
