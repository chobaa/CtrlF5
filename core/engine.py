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

def get_rag_chain(vectorstore, tech_stack: str = "General", is_strict: bool = True, relevance_threshold: float = 0.0, top_k: int = 5):
    """
    RAG 파이프라인 생성 (Ensemble + Rerank + LLM)
    """
    # 1. 캐시된 모델 로드
    llm = load_llm()
    reranker_model = load_reranker_model()
    
    # --- [STEP 1] 1차 검색: Ensemble (Vector + BM25) ---
    # Top-K보다 조금 더 많이 가져와서 Reranking (보통 2~3배수)
    fetch_k = max(20, top_k * 3)
    vector_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": fetch_k})
    base_retriever = vector_retriever
    
    try:
        collection_data = vectorstore.get() 
        texts = collection_data['documents']
        metadatas = collection_data['metadatas']
        
        if texts:
            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas) if t]
            if docs:
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = fetch_k
                base_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
    except Exception as e:
        print(f"⚠️ BM25 생성 실패: {e}")

    # --- [STEP 2] 2차 검색: Reranker (Manual Implementation) ---
    try:
        # User defined Top-K used here
        compressor = CrossEncoderReranker(model=reranker_model, top_n=top_k)
        
        # ContextualCompressionRetriever 대신 수동 실행 (점수 보존 확인용)
        def manual_rerank_retrieval(query: str):
            # 1. Base Retrieval
            initial_docs = base_retriever.invoke(query)
            
            if not initial_docs:
                return []
            
            # 2. Score Calculation (Directly using the model)
            # HuggingFaceCrossEncoder (LangChain wrapper) usually has a 'score' method causing issues or stripping results.
            # We construct pairs manually.
            pairs = [[query, doc.page_content] for doc in initial_docs]
            
            try:
                # Try accessing the underlying sentence-transformer model directly if possible
                # reranker_model is HuggingFaceCrossEncoder
                # Its logic: scores = model.predict(pairs)
                scores = reranker_model.model.predict(pairs)
                
                # Check if scores are a list of floats or tensor
                if hasattr(scores, 'tolist'):
                    scores = scores.tolist()
            except Exception as e:
                print(f"⚠️ Direct scoring failed, trying LangChain scoring: {e}")
                # Fallback to LangChain 'score' method if available
                scores = reranker_model.score(pairs)

            # 3. Attach scores and sort (with Sigmoid Normalization)
            import math
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))

            docs_with_scores = []
            for doc, score in zip(initial_docs, scores):
                # score가 array/tensor일 경우 float로 변환
                if isinstance(score, list): score = score[0]
                
                # Sigmoid 적용하여 0~1 사이 확률값으로 변환
                normalized_score = sigmoid(float(score))
                
                doc.metadata['relevance_score'] = normalized_score
                # 원본 점수도 저장 (디버깅용)
                doc.metadata['raw_score'] = float(score)
                
                docs_with_scores.append(doc)
            
            # 4. Sort by score (descending)
            docs_with_scores.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
            
            # 5. Top-K Slice
            final_docs = docs_with_scores[:top_k]
            
            # DEBUG
            if final_docs:
                print(f"🔎 [DEBUG] Top-1 Score: {final_docs[0].metadata['relevance_score']}")
            
            return final_docs

        from langchain_core.runnables import RunnableLambda
        retriever = RunnableLambda(manual_rerank_retrieval)

    except Exception as e:
        print(f"⚠️ Reranker 로드 실패: {e}")
        retriever = base_retriever

    # --- [STEP 3] Filtering with Threshold ---
    def filter_documents(docs: list[Document]) -> list[Document]:
        """점수 기반 문서 필터링"""
        if not docs:
            return []
        
        # DEBUG: 첫 번째 문서의 메타데이터 확인
        if docs:
            print(f"🔎 First Doc Metadata keys: {docs[0].metadata.keys()}")
            print(f"🔎 First Doc Metadata score: {docs[0].metadata.get('relevance_score')}")

        filtered = []
        for doc in docs:
            # metadata 키 확인 (relevance_score 또는 score)
            score = doc.metadata.get("relevance_score", doc.metadata.get("score", None))
            
            # 점수가 없거나, 임계값보다 높은 경우만 유지
            if score is None:
                # 점수가 없으면 필터링하지 않고 유지 (안전장치)
                filtered.append(doc)
            elif score >= relevance_threshold:
                # 점수를 metadata에 명시적으로 'relevance_score'로 통일 (UI 표시용)
                doc.metadata['relevance_score'] = score
                filtered.append(doc)
                
        return filtered

    from langchain_core.runnables import RunnableLambda
    filtered_retriever = retriever | RunnableLambda(filter_documents)

    # 4. 질문 재구성 (History Aware)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question..."
        "(Do NOT answer the question, just reformulate it)"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # retriever -> filtered_retriever 변경
    history_aware_retriever = create_history_aware_retriever(
        llm, filtered_retriever, contextualize_q_prompt
    )

    # 5. 답변 생성 (QA Chain)
    system_instruction = get_system_prompt(tech_stack, is_strict)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain