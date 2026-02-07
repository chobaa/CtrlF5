from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# [LangChain 1.0 호환] Classic 모듈 사용
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# [고급 검색] Ensemble & BM25
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def get_rag_chain(vectorstore, tech_stack="General Development"):
    # 1. 모델 설정
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    # 2. 고급 검색기 설정 (Ensemble: Vector + BM25)
    # 2-1. 벡터 검색기 (MMR)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5}
    )
    
    # 2-2. BM25 검색기 생성 및 앙상블
    try:
        # ChromaDB에서 데이터 추출
        collection_data = vectorstore.get() 
        texts = collection_data['documents']
        metadatas = collection_data['metadatas']
        
        if texts:
            # Document 객체로 변환
            docs = [
                Document(page_content=t, metadata=m) 
                for t, m in zip(texts, metadatas) if t
            ]
            
            if docs:
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 5
                
                # 앙상블 검색기 (Vector 50% + BM25 50%)
                retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
            else:
                retriever = vector_retriever
        else:
            retriever = vector_retriever
            
    except Exception as e:
        print(f"⚠️ BM25 생성 실패 (기본 검색 사용): {e}")
        retriever = vector_retriever

    # 3. 질문 재구성 (Contextualize)
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

    # 4. 답변 생성 프롬프트 (RAG 우선 순위 강화)
    qa_system_prompt = (
        f"You are 'Ctrl + F5', a world-class expert developer in {tech_stack}. "
        f"Your goal is to provide structured, easy-to-read answers about {tech_stack}. "
        "\n\n"
        "🔥 CRITICAL RULES:\n"
        "1. **Answer in Korean** (unless the user asks in English).\n"
        "2. **SOURCE OF TRUTH = CONTEXT**: Your internal knowledge might be outdated. "
        "**ALWAYS prioritize the provided Context.** If the Context contradicts your internal knowledge, "
        "you MUST follow the Context. Do NOT use deprecated syntax unless it is explicitly shown in the Context.\n"
        f"3. Strictly follow the latest syntax of {tech_stack} based on the Context.\n"
        "4. If the answer is not in the context, politely say that the information is not in the provided documents.\n"
        "\n"
        "🎨 FORMATTING RULES (STRICTLY FOLLOW THIS STRUCTURE):\n"
        "1. **Start with a Summary**: Briefly explain the concept or solution in 1-2 sentences.\n"
        "2. **Code Block**: Provide the complete code in a single, copyable code block.\n"
        "   - Use correct syntax highlighting (e.g., ```python, ```java).\n"
        "   - Include all necessary imports.\n"
        "3. **Key Points**: Use a bulleted list to explain the critical parts of the code.\n"
        "4. **Separation**: Do NOT mix code and explanation in the same paragraph. Keep them visually distinct.\n"
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