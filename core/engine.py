from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def get_rag_chain(vectorstore, tech_stack="General Development"):
    # 1. 모델 설정
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    # 2. 검색기 설정
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # 3. 질문 재구성 (Contextualize Question)
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

    # 4. 답변 생성 (구조화된 프롬프트 적용)
    qa_system_prompt = (
        f"You are 'Ctrl + F5', a world-class expert developer in {tech_stack}. "
        f"Your goal is to provide structured, easy-to-read answers about {tech_stack}. "
        "\n\n"
        "🔥 CRITICAL RULES:\n"
        "1. **Answer in Korean** (unless the user asks in English).\n"
        f"2. Base your answer ONLY on the provided context.\n"
        f"3. Strictly follow the latest syntax of {tech_stack}.\n"
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