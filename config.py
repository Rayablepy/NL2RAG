import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

#chat model name
CHAT_MODEL_NAME=os.getenv("CHAT_MODEL_NAME")
#url to listen to for transformers
BASE_URL=os.getenv("BASE_URL")
MODEL_API_KEY='not-needed'
#cache dir for models
MODEL_PATH="./NL2RAG_models/"
#file path for uploaded files
ACTUAL_FILE_PATH="./user_data/"
#embedding model configs
EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL_CONTEXT=int(os.getenv("EMBEDDING_MODEL_CONTEXT", 512))
EMBEDDING_MODEL_CHUNK=int(os.getenv("EMBEDDING_MODEL_CHUNK", 64))
#reranker model configs
RERANKER_MODEL_NAME=os.getenv("RERANKER_MODEL_NAME")

#2-step agentic rag configs
RELEVANCE_THRESHOLD = 0.55
REWRITE_BUFFER=3
MAX_STEPS=15

CHAT_MODEL=ChatOpenAI(
    model=CHAT_MODEL_NAME,
    base_url=BASE_URL,
    api_key=MODEL_API_KEY,
    temperature=0
)
