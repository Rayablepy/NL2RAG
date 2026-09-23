import os
import textract
from functools import lru_cache
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from config import ACTUAL_FILE_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_CONTEXT, EMBEDDING_MODEL_CHUNK
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )


@lru_cache(maxsize=1)
def get_store():
    return Chroma(
        collection_name="NL2SQL",
        embedding_function=get_embeddings(),
        persist_directory="./chroma_NL2RAG",
    )


@lru_cache(maxsize=1)
def get_retriever():
    return get_store().as_retriever(
        search_type="hybrid",
        search_kwargs={"k": 5},
    )


@lru_cache(maxsize=1)
def get_text_splitter():
    return TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=EMBEDDING_MODEL_CONTEXT,
        chunk_overlap=EMBEDDING_MODEL_CHUNK,
    )


def read_data(file_path: str) -> list[Document]:
    try:
        text = textract.process(file_path).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")
    return [
        Document(
            page_content=text,
            metadata={"source": os.path.basename(file_path)},
        )
    ]


def save_data(file_name: str, batch_size: int = 50):
    full_path = os.path.join(ACTUAL_FILE_PATH, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    source = os.path.basename(full_path)
    get_store().delete(where={"source": source})
    docs = read_data(full_path)
    splits = get_text_splitter().split_documents(docs)
    get_store().add_documents(documents=splits, batch_size=batch_size)


def list_data() -> list[str]:
    collection = get_store()._collection
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results.get("metadatas", []) or []:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sorted(sources)


def delete_data(file_name: str) -> str:
    source = os.path.basename(file_name)
    get_store().delete(where={"source": source})
    return f"Deleted documents with source: {source}"


@tool
async def query_data(query: str) -> str:
    """Query a local RAG database for information matching the query

    Args:
        query (str): The query string to search for in the database

    Returns:
        str: The matching results from the database
    """
    results = await get_retriever().ainvoke(query)
    lines = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content[:2000]
        lines.append(f"[source: {source}]\n{content}")
    return "Results:\n\n" + "\n".join(lines)
