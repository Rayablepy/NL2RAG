import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

CHAT_MODEL_NAME=os.getenv("CHAT_MODEL_NAME")
BASE_URL=os.getenv("BASE_URL")
MODEL_API_KEY=os.getenv("MODEL_API_KEY")
MODEL_PATH="./NL2RAG_models/"
ACTUAL_FILE_PATH="./user_data/"
EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL_CONTEXT=int(os.getenv("EMBEDDING_MODEL_CONTEXT", 512))
EMBEDDING_MODEL_CHUNK=int(os.getenv("EMBEDDING_MODEL_CHUNK", 64))

CHAT_MODEL=ChatOpenAI(
    model=CHAT_MODEL_NAME,
    base_url=BASE_URL,
    api_key=MODEL_API_KEY,
)
