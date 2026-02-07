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
    # [모델 변경] 영어 전용(bge-large-en) -> 다국어 지원(bge-m3)
    # 한글 질문으로 영어 문서를 검색할 때 성능이 훨씬 좋습니다.
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"LangChain": [], "Spring Boot": [], "React": []}

def load_vectorstore(stack_name: str):
    collection = stack_name.replace(" ", "_").lower()
    return Chroma(
        persist_directory=DB_PATH, 
        embedding_function=get_embeddings(),
        collection_name=collection
    )

def fetch_url_content(url: str):
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
    if not docs: return None

    # 언어별 최적화
    lang = Language.PYTHON
    if "Spring" in stack_name or "Java" in stack_name: lang = Language.JAVA
    elif any(x in stack_name for x in ["React", "JS", "Node"]): lang = Language.JS
    elif "Go" in stack_name: lang = Language.GO
    
    # 청크 사이즈 800 유지
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang, chunk_size=800, chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    
    collection = stack_name.replace(" ", "_").lower()
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=get_embeddings(), 
        persist_directory=DB_PATH,
        collection_name=collection
    )
    return vectorstore