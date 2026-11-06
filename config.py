import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _get_int(name: str, default: int) -> int:
    """
    Read an integer setting from the environment.

    Everything in a `.env` file is text, so "400" arrives as the string "400".
    This helper converts it to a real integer and falls back to the default if
    the variable is missing or is not a valid number.

    Args:
        name: Environment variable name, for example "CHILD_CHUNK_SIZE".
        default: Value to use when the variable is missing or invalid.

    Returns:
        The setting as an integer.
    """    
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        print(f"[config] '{name}={raw_value}' is not a number. Using the value -> {default}")
        return default

# folders
PROJECT_ROOT = Path(__file__).parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_DIR = VECTORSTORE_DIR / "faiss"

DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"

# openai models
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LLM_TEMPERATURE = 0.0


# faiss
VECTOR_DB = os.getenv("VECTOR_DB", "faiss").strip().lower()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# chunking
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# retrieval pipeline
NUM_QUERY_VARIENTS = _get_int("NUM_QUERY_VARIENTS", 3)

VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)

HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)

RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)

MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))

MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)

RERANKER = os.getenv("RERANKER", "cross-encoder").strip().lower()

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)

def check_groq_key() -> None:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )


def check_pinecone_config() -> None:
    if not PINECONE_API_KEY:
        raise ValueError(
            "VECTOR_DB=pinecone but PINECONE_API_KEY is empty."
        )

    if not PINECONE_INDEX_NAME:
        raise ValueError(
            "VECTOR_DB=pinecone but PINECONE_INDEX_NAME is empty."
        )


def describe_settings() -> dict:
    return {
        "Chat Model": GROQ_CHAT_MODEL,
        "Embedding Model": EMBEDDING_MODEL,
        "VectorDB": VECTOR_DB,
        "Parent Chunk Size": PARENT_CHUNK_SIZE,
        "Child Chunk Size": CHILD_CHUNK_SIZE,
    }