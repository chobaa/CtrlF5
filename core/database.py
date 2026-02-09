# core/database.py
import os
import json
import bs4
from typing import List, Optional, Dict
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 캐시된 모델 로더 사용
from core.model_loader import load_embedding_model

DB_PATH = "./chroma_db_expert"
CONFIG_PATH = "stacks_config.json"

def save_config(config: Dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def load_config() -> Dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"LangChain": [], "Spring Boot": [], "React": []}

def load_vectorstore(stack_name: str) -> Chroma:
    collection = stack_name.replace(" ", "_").lower()
    return Chroma(
        persist_directory=DB_PATH, 
        embedding_function=load_embedding_model(),
        collection_name=collection
    )

def fetch_url_content(url: str) -> List[Document]:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            header_template=headers,
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(["article", "main", "div", "p"]))
        )
        return loader.load()
    except Exception as e:
        print(f"❌ Error loading {url}: {e}")
        return []

def build_vectorstore(docs: List[Document], stack_name: str) -> Optional[Chroma]:
    if not docs: return None

    # 언어별 최적화
    lang = Language.PYTHON
    if "Spring" in stack_name or "Java" in stack_name: lang = Language.JAVA
    elif any(x in stack_name for x in ["React", "JS", "Node"]): lang = Language.JS
    elif "Go" in stack_name: lang = Language.GO
    
    # 청크 사이즈 최적화 (800)
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang, chunk_size=800, chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    
    collection = stack_name.replace(" ", "_").lower()
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=load_embedding_model(), 
        persist_directory=DB_PATH,
        collection_name=collection
    )
    return vectorstore