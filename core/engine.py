# core/engine.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# LangChain Modules
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# Internal Modules
from core.model_loader import load_llm, load_reranker_model
from core.prompts import get_system_prompt

def get_rag_chain(vectorstore, tech_stack: str = "General", is_strict: bool = True):
    """
    RAG 파이프라인 생성 (Ensemble + Rerank + LLM)
    """
    # 1. 캐시된 모델 로드
    llm = load_llm()
    reranker_model = load_reranker_model()
    
    # --- [STEP 1] 1차 검색: Ensemble (Vector + BM25) ---
    vector_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15})
    base_retriever = vector_retriever
    
    try:
        collection_data = vectorstore.get() 
        texts = collection_data['documents']
        metadatas = collection_data['metadatas']
        
        if texts:
            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas) if t]
            if docs:
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 15
                base_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
    except Exception as e:
        print(f"⚠️ BM25 생성 실패: {e}")

    # --- [STEP 2] 2차 검색: Reranker ---
    try:
        top_n = 5 if is_strict else 8
        compressor = CrossEncoderReranker(model=reranker_model, top_n=top_n)
        
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    except Exception as e:
        print(f"⚠️ Reranker 로드 실패: {e}")
        retriever = base_retriever

    # 3. 질문 재구성 (History Aware)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question..."
        "(Do NOT answer the question, just reformulate it)"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. 답변 생성 (QA Chain)
    system_instruction = get_system_prompt(tech_stack, is_strict)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain