import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

CHAT_MODEL_NAME=os.getenv("CHAT_MODEL_NAME")
ACTUAL_FILE_PATH="./user_data/"
EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_MODEL_CONTEXT=int(os.getenv("EMBEDDING_MODEL_CONTEXT", 512))
EMBEDDING_MODEL_CHUNK=int(os.getenv("EMBEDDING_MODEL_CHUNK", 64))

tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
hf = HuggingFacePipeline(pipeline=pipe)
model=ChatHuggingFace(llm=hf)
