import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import streamlit as st
import tempfile
import os 

from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import shutil
import time
import boto3
# 페이지 설정
st.set_page_config(
    page_title="상품 문의 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# S3 관련 설정 (직접 secrets에서 가져오기)
try:
    BUCKET_NAME = st.secrets.S3_BUCKET_NAME
    S3_DB_FOLDER = st.secrets.S3_DB_FOLDER
    AWS_ACCESS_KEY_ID = st.secrets.AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY = st.secrets.AWS_SECRET_ACCESS_KEY
    AWS_REGION = st.secrets.AWS_REGION
except Exception as e:
    st.error("Secrets 설정을 확인해주세요.")
    st.stop()

def get_aws_session():
    return boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
def get_bedrock_client():
    session = get_aws_session()
    return session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2"
    )

def download_db_from_s3(bucket_name: str, s3_folder: str, local_path: str):
    """S3에서 ChromaDB 파일들을 다운로드"""
    session = get_aws_session()
    s3_client = session.client('s3')
    os.makedirs(local_path, exist_ok=True)  # Use os.makedirs
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)
        
        for page in pages:
            for obj in page.get('Contents', []):
                relative_path = obj['Key'][len(s3_folder):].lstrip('/')
                if not relative_path:
                    continue
                    
                local_file_path = os.path.join(local_path, relative_path)  # Use os.path.join
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)  # Use os.makedirs
                
                s3_client.download_file(
                    bucket_name,
                    obj['Key'],
                    local_file_path
                )
    except Exception as e:
        raise Exception(f"S3에서 DB 다운로드 중 오류 발생: {str(e)}")

def get_current_datetime_with_day():
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    weekday = now.strftime("%A")
    return f"{year}년{month}월{day}일{weekday} {hour}시{minute}분"

def load_chroma_db(base_path: str):
    """Chroma DB 로드"""
    if not os.path.exists(base_path):
        raise ValueError(f"데이터베이스가 존재하지 않습니다: {base_path}")
    
    try:
        bedrock_runtime = get_bedrock_client()
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            client=bedrock_runtime
        )
        
        # 모든 하위 디렉토리에 대한 권한 확인
        for root, dirs, files in os.walk(base_path):
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, 0o777)
            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, 0o777)
        
        db = Chroma(
            persist_directory=base_path,
            embedding_function=embeddings,
        )
        
        # 연결 테스트
        db._collection.count()
        
        return db
        
    except Exception as e:
        print(f"ChromaDB 로드 오류: {str(e)}")
        raise Exception(f"ChromaDB 로드 실패: {str(e)}")
def get_product_info_from_db(db: Chroma):
    """Chroma DB에서 제품 정보 가져오기"""
    try:
        collection = db._collection
        metadatas = collection.get()['metadatas']
        
        product_info = {}
        for metadata in metadatas:
            if metadata and 'product_uuid' in metadata and 'product_name' in metadata:
                product_uuid = metadata['product_uuid']
                product_name = metadata['product_name']
                if product_uuid not in product_info:
                    product_info[product_uuid] = product_name
        
        return product_info
    except Exception as e:
        st.sidebar.error(f"제품 정보 조회 중 오류 발생: {str(e)}")
        return {}

def clear_chat_history():
    """채팅 관련 세션 상태를 초기화하는 함수"""
    keys_to_delete = ['messages', 'conversation_chain', 'current_product_id']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

def retrieve_docs(query: str, db=None, product_uuid=None):
    """문서 검색 함수"""
    try:
        if db is None or product_uuid is None:
            return []
        
        return db.similarity_search(
            query,
            k=3,
            filter={"product_uuid": product_uuid}
        )
    except Exception as e:
        print(f"문서 검색 중 오류 발생: {str(e)}")
        return []
    
def create_rag_chain(db: Chroma, product_uuid: str):
    """RAG 체인 생성"""
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        client=get_bedrock_client(),
        model_kwargs={
            "temperature": 0,
            "max_tokens": 4000
        }
    )
    
    # 컨텍스트 인식 질문 재구성을 위한 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # 메인 QA 프롬프트
    current_time = get_current_datetime_with_day()
    qa_system_prompt = f"""당신은 이 상품의 판매자입니다.
    중요 규칙:
    1. 컨텍스트에 명시된 정보만 사용해야 합니다
    - 컨텍스트에 없는 내용은 "해당 정보가 없어 답변드리기 어렵습니다"라고 답변하세요
    - 추측이나 일반적인 답변은 금지됩니다
    2. 답변 전 반드시 컨텍스트를 확인하세요.
    - 가격, 수량, 배송일정 등 수치는 반드시 컨텍스트와 일치해야 함
    - 없는 정보를 임의로 생성하지 마세요
    3. 질문과 직접 관련된 정보만 답변하세요
    - 불필요한 부가 설명이나 맥락은 제외
    - 이전 대화에서 확인된 내용만 참조
    4. 배송 관련 질문에 공휴일과 현재 시간: '{current_time}'을 고려하세요
    
    답변을 3가지 스타일로 작성하고 각 답변 사이에 ### 를 넣어서 구분해주세요.
    1번째 답변: 문의내역 + 동일한 스타일
    2번째 답변: 문의내역 + 창의적인 스타일
    3번째 답변: 문의내역 + 재밌는 답변

    형식: 답변1 ###\n 답변2 ###\n 답변3

    주의사항:
    1. 오직 답변과 구분자(###)만 출력하세요
    2. 컨텍스트의 대화 스타일과 이모지를 참고하세요
    3. 절대 고객의 개인 정보(주소,휴대폰 번호 등)는 답변하지마세요.

    상품 정보 및 과거 문의 내역:
    {{context}}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # 검색기 설정
    retriever = db.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"product_uuid": product_uuid}
        }
    )

    # History-aware retriever 생성
    def create_history_aware_retriever(llm, retriever, prompt):
        def get_retriever_chain(llm, prompt):
            chain = prompt | llm | StrOutputParser()
            return chain

        def historical_retriever(inputs):
            question = inputs["question"]
            chat_history = inputs["chat_history"]
            
            context_chain = get_retriever_chain(llm, prompt)
            contextualized_q = context_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            
            return retriever.invoke(contextualized_q)

        return historical_retriever

    # Document chain 생성
    def create_stuff_documents_chain(llm, prompt):
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(x["documents"])
            ) 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        return chain

    # 전체 체인 조합
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: x["chat_history"]
        ) 
        | {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "documents": lambda x: history_aware_retriever({
                "question": x["question"],
                "chat_history": x["chat_history"]
            })
        } 
        | doc_chain
    )

    return chain


        
def main():
    st.title("상품 문의 챗봇 🤖")
    
    # 세션 상태 초기화
    if 'db' not in st.session_state:
        try:
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp(prefix="chromadb_")
            
            # S3에서 DB 다운로드
            with st.spinner("데이터베이스를 불러오는 중..."):
                download_db_from_s3(BUCKET_NAME, S3_DB_FOLDER, temp_dir)
            
            # 임시 디렉토리의 모든 파일에 대한 권한 설정
            for root, dirs, files in os.walk(temp_dir):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o777)
            
            # ChromaDB 로드
            db = load_chroma_db(temp_dir)
            
            # 세션 상태에 저장
            st.session_state.db = db
            st.session_state.temp_dir = temp_dir
            
        except Exception as e:
            st.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return
    
    try:
        # 저장된 DB 인스턴스 사용
        db = st.session_state.db
        product_info = get_product_info_from_db(db)
        
        if not product_info:
            st.error("제품 정보를 불러올 수 없습니다. 데이터베이스를 확인해주세요.")
            return
        
        # 사이드바 설정
        with st.sidebar:
            st.title("제품 선택 🛍️")
            st.markdown("---")
            
            product_names = {name: uuid for uuid, name in product_info.items()}
            selected_name = st.selectbox(
                "문의하실 제품을 선택하세요",
                options=list(product_names.keys()),
                key="product_selector"
            )
            selected_product_id = product_names[selected_name]
            
            st.markdown("---")
            
            if st.button("💫 채팅 기록 초기화", use_container_width=True):
                clear_chat_history()
                st.success("채팅 기록이 초기화되었습니다!")
                st.rerun()
            
            st.markdown("### 선택된 제품 정보")
            st.info(f"현재 선택: {selected_name}")
        
        # 세션 상태 초기화 또는 업데이트
        if ('conversation_chain' not in st.session_state or 
            'current_product_id' not in st.session_state or 
            st.session_state.current_product_id != selected_product_id):
            
            # 제품이 변경되었을 때 채팅 기록 초기화
            if ('current_product_id' in st.session_state and 
                st.session_state.current_product_id != selected_product_id):
                clear_chat_history()
            
            chain = create_rag_chain(db, selected_product_id)
            st.session_state.conversation_chain = chain
            st.session_state.current_product_id = selected_product_id
    
            # 페이지 새로고침
            st.rerun()
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # 초기 메시지 표시
        if not st.session_state.messages:
            st.markdown(f"""
            ### 👋 안녕하세요!
            **{product_info[selected_product_id]}**에 대해 궁금하신 점을 자유롭게 문의해주세요.
            """)
        
        # 이전 메시지들 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 새로운 사용자 입력 처리
        if prompt := st.chat_input("무엇을 도와드릴까요?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner('답변을 생성중입니다...'):
                    try:
                        # 응답 생성
                        response = st.session_state.conversation_chain.invoke({
                            "question": prompt,
                            "chat_history": st.session_state.messages
                        })

                        if not response:
                            raise ValueError("응답이 생성되지 않았습니다.")

                        # 응답을 ### 구분자로 분리하고 처리
                        answers = response.split("###")
                        answers = [answer.strip() for answer in answers]
                        
                        # 필요한 경우 답변 리스트 보충
                        while len(answers) < 3:
                            answers.append("답변이 없습니다.")
                        
                        # 참고 문서 가져오기
                        docs = retrieve_docs(prompt, db, selected_product_id)
                        
                        # 탭으로 다양한 응답 스타일 표시
                        tab1, tab2, tab3 = st.tabs(["💬 기본 답변", "✨ 창의적 답변", "😊 재미있는 답변"])
                        
                        with tab1:
                            st.markdown(answers[0])
                            if docs and len(docs) > 0:
                                with st.expander("참고 문서 1"):
                                    st.markdown(f"**내용:**\n{docs[0].page_content}")
                                    st.markdown("**메타데이터:**")
                                    st.json(docs[0].metadata)
                        
                        with tab2:
                            st.markdown(answers[1])
                            if docs and len(docs) > 1:
                                with st.expander("참고 문서 2"):
                                    st.markdown(f"**내용:**\n{docs[1].page_content}")
                                    st.markdown("**메타데이터:**")
                                    st.json(docs[1].metadata)
                        
                        with tab3:
                            st.markdown(answers[2])
                            if docs and len(docs) > 2:
                                with st.expander("참고 문서 3"):
                                    st.markdown(f"**내용:**\n{docs[2].page_content}")
                                    st.markdown("**메타데이터:**")
                                    st.json(docs[2].metadata)
                        
                        # 첫 번째 답변을 채팅 히스토리에 저장
                        st.session_state.messages.append(

                            {"role": "assistant", "content": answers[0]}
                        )
                        
                    except Exception as e:
                        st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")
                        return

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        return
        
    finally:
        # 세션이 리셋될 때만 임시 디렉토리 정리
        if st.session_state.get('_is_new_session', True):
            try:
                temp_dir = st.session_state.get('temp_dir')
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                st.warning(f"임시 파일 정리 중 오류 발생: {cleanup_error}")
            finally:
                st.session_state['_is_new_session'] = False

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        st.error("자세한 오류 정보:", exc_info=True)
