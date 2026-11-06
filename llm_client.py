from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGroq:
    """
    Create (or reuse) the Groq chat model used for every LLM call.

    Returns:
        A configured LangChain ChatGroq model.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
    """
    config.check_groq_key()

    return ChatGroq(
        model=config.GROQ_CHAT_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create (or reuse) the local HuggingFace embedding model.

    Returns:
        A configured HuggingFaceEmbeddings model.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
