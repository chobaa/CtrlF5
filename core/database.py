import os
import json
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = "./chroma_db_expert"
CONFIG_PATH = "stacks_config.json"

def get_embeddings():
    # 로컬 CPU에서도 잘 돌아가는 고성능 임베딩 모델
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def save_config(config):
    """설정(스택 및 URL 리스트)을 JSON으로 저장"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def load_config():
    """저장된 설정을 불러옴"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"LangChain": [], "Spring Boot": [], "React": []}

def load_vectorstore(stack_name: str):
    """특정 스택의 DB 로드"""
    collection = stack_name.replace(" ", "_").lower()
    return Chroma(
        persist_directory=DB_PATH, 
        embedding_function=get_embeddings(),
        collection_name=collection
    )

def fetch_url_content(url: str):
    """
    [UI용] 단일 URL 내용을 가져옵니다. 
    Progress Bar 업데이트를 위해 분리했습니다.
    """
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

def build_vectorstore(docs: list, stack_name: str):
    """
    [Core] 모인 문서들을 분할하고 임베딩하여 저장합니다.
    """
    if not docs: return None

    # 언어별 최적화 (코드 분할 성능 향상)
    lang = Language.PYTHON
    if "Spring" in stack_name or "Java" in stack_name: lang = Language.JAVA
    elif any(x in stack_name for x in ["React", "JS", "Node"]): lang = Language.JS
    elif "Go" in stack_name: lang = Language.GO
    
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang, chunk_size=1500, chunk_overlap=300
    )
    splits = text_splitter.split_documents(docs)
    
    collection = stack_name.replace(" ", "_").lower()
    
    # Chroma DB 저장
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=get_embeddings(), 
        persist_directory=DB_PATH,
        collection_name=collection
    )
    return vectorstore