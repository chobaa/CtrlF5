from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# [LangChain 1.0 호환] Classic 모듈
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# [고급 검색] Ensemble, BM25, ContextualCompression
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

def get_rag_chain(vectorstore, tech_stack="General Development", is_strict=True):
    # 1. 모델 설정
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    # --- [STEP 1] 1차 검색: Ensemble (Vector + BM25) ---
    # 후보군을 넉넉하게(15개) 가져옵니다.
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

    # --- [STEP 2] 2차 검색: Reranker (모델 교체!) ---
    try:
        # [핵심 변경] 영어 전용 모델 -> 다국어(한글) 지원 모델로 교체
        # BAAI/bge-reranker-v2-m3 : 한글/영어 모두 이해도가 매우 높은 SOTA 모델
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        
        # Strict 모드일 때 5개, 아니면 8개 선택
        top_n = 5 if is_strict else 8
        
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    except Exception as e:
        print(f"⚠️ Reranker 로드 실패: {e}")
        retriever = base_retriever

    # 3. 질문 재구성
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. 답변 생성 프롬프트
    if is_strict:
        system_instruction = (
            "🔥 CRITICAL RULES (STRICT MODE ON):\n"
            "1. **SOURCE OF TRUTH = CONTEXT**: You must prioritize the provided Context above all else.\n"
            "2. **NO LAZY REFERRALS**: Do NOT say 'check the docs'. Extract and explain the details directly from the Context.\n"
            "3. **DERIVE FROM CONTEXT**: You are allowed to synthesize and infer the answer if the logic is supported by the Context.\n"
            "4. **FAIL CONDITION**: Only if the Context provides **NO relevant information** to derive an answer, then respond exactly with:\n"
            "   '⚠️ 제공된 문서에서 해당 내용에 대한 근거를 찾을 수 없습니다.'\n"
        )
    else:
        system_instruction = (
            "⚠️ RULES (STRICT MODE OFF):\n"
            "1. **Prioritize Context**: Check the provided Context first.\n"
            "2. **Internal Knowledge Allowed**: If Context is missing, you MAY use your internal knowledge.\n"
        )

    qa_system_prompt = (
        f"You are 'Ctrl + F5', a world-class expert developer in {tech_stack}. "
        f"{system_instruction}"
        "\n"
        "🎨 FORMATTING RULES:\n"
        "1. **Answer in Korean**.\n"
        "2. **Start with a Summary**: 1-2 sentences.\n"
        "3. **Code Block**: Complete code with imports.\n"
        f"4. Strictly follow the latest syntax of {tech_stack}.\n"
        "\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain