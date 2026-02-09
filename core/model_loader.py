# core/model_loader.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

@st.cache_resource
def load_llm(model_name: str = "gpt-5-nano"):
    """LLM 모델 로딩 및 캐싱"""
    return ChatOpenAI(model=model_name, temperature=0)

@st.cache_resource
def load_embedding_model():
    """임베딩 모델(BGE-M3) 로딩 및 캐싱 (CPU/GPU 메모리 절약)"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_reranker_model():
    """리랭커 모델(BGE-Reranker-M3) 로딩 및 캐싱"""
    return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")