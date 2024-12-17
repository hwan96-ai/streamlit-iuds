__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import tempfile
import os 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
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

# def load_chroma_db(base_path: str):
#     """Chroma DB 로드"""
#     if not os.path.exists(base_path):  # os.path 사용
#         raise ValueError(f"데이터베이스가 존재하지 않습니다: {base_path}")
    
#     try:
#         bedrock_runtime = get_bedrock_client()
#         embeddings = BedrockEmbeddings(
#             model_id="amazon.titan-embed-text-v1",
#             client=bedrock_runtime
#         )
        
#         db = Chroma(
#             persist_directory=base_path,
#             embedding_function=embeddings,
#         )
#         return db
#     except Exception as e:
#         raise Exception(f"ChromaDB 로드 실패: {str(e)}")
def load_chroma_db(base_path: str):
    """Chroma DB 로드"""
    try:
        if not os.path.exists(base_path):
            raise ValueError(f"데이터베이스가 존재하지 않습니다: {base_path}")
        
        # Bedrock 클라이언트 설정
        bedrock_runtime = get_bedrock_client()
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            client=bedrock_runtime
        )
        
        # ChromaDB 초기화
        db = Chroma(
            persist_directory=base_path,
            embedding_function=embeddings,
            collection_name="product_qa"  # 컬렉션 이름 지정
        )
        return db
        
    except Exception as e:
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
    
def create_filtered_retriever(db: Chroma, product_uuid: str):
    """특정 product_uuid에 대한 필터링된 retriever 생성"""
    search_kwargs = {
        "filter": {'product_uuid': product_uuid},
        "k": 3
    }
    
    return db.as_retriever(
        search_kwargs=search_kwargs,
        search_type="similarity"
    )

def create_rag_chain(db: Chroma, product_uuid: str):
    """RAG 체인 생성"""
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        client=get_bedrock_client(),
        model_kwargs={
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "max_tokens": 4000,
            "stop_sequences": ["\n\nHuman"]
        },
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    # 필터링된 retriever 생성
    retriever = create_filtered_retriever(db, product_uuid)

    # 컨텍스트 인식 질문 재구성
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # History-aware retriever 생성
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA 체인 생성
    qa_system_prompt = """당신은 이 상품의 판매자입니다.
    중요 규칙:
    1. 컨텍스트에 명시된 정보만 사용해야 합니다
    - 컨텍스트에 없는 내용은 "해당 정보가 없어 답변드리기 어렵습니다"라고 답변하세요
    - 추측이나 일반적인 답변은 금지됩니다
    2. 답변 전 반드시 컨텍스트를 확인하세요.
    3. 질문과 직접 관련된 정보만 답변하세요
    Context: {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Document chain 생성
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 최종 RAG 체인 생성
    rag_chain = create_retrieval_chain(
        history_aware_retriever, 
        question_answer_chain
    )

    return rag_chain

        
def main():
    st.title("상품 문의 챗봇 🤖")
    
    # 현재 작업 디렉토리 내에 임시 디렉토리 생성
    temp_dir = os.path.join(os.getcwd(), "temp_chroma_db")
    os.makedirs(temp_dir, exist_ok=True)
    os.chmod(temp_dir, 0o777)
    
    db = None
    
    try:
        # S3에서 DB 다운로드
        with st.spinner("데이터베이스를 불러오는 중..."):
            try:
                download_db_from_s3(BUCKET_NAME, S3_DB_FOLDER, temp_dir)
                st.success("데이터베이스 다운로드 완료")
                
                # 디버깅 정보 출력
                st.write("임시 디렉토리 경로:", temp_dir)
                st.write("디렉토리 내용:", os.listdir(temp_dir))
                
                # 파일 권한 확인
                for root, dirs, files in os.walk(temp_dir):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o666)
                
            except Exception as e:
                st.error(f"데이터베이스 다운로드 실패: {str(e)}")
                return
        
        # DB 로드 시도
        try:
            db = load_chroma_db(temp_dir)
            st.success("데이터베이스 로드 완료")
            
            # 제품 정보 가져오기
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
                                "input": prompt,
                                "chat_history": st.session_state.messages
                            })

                            if not response.get('answer'):
                                raise ValueError("응답이 생성되지 않았습니다.")

                            # 응답 처리
                            answer = response['answer']
                            source_documents = response.get('source_documents', [])
                            
                            # 응답 표시
                            st.write(answer)
                            
                            # 참고 문서 표시
                            if source_documents:
                                with st.expander("참고 문서"):
                                    for i, doc in enumerate(source_documents[:3], 1):
                                        st.markdown(f"**문서 {i}**")
                                        st.markdown(f"내용: {doc.page_content}")
                                        st.markdown("메타데이터:")
                                        st.json(doc.metadata)
                            
                            # 응답을 채팅 히스토리에 저장
                            st.session_state.messages.append(
                                {"role": "assistant", "content": answer}
                            )
                            
                        except Exception as e:
                            st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")
                            return
                            
        except Exception as e:
            st.error(f"데이터베이스 로드 실패: {str(e)}")
            return
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        return
        
    finally:
        # 리소스 정리
        if db is not None:
            try:
                if hasattr(db, '_collection'):
                    db._collection.count()
            except Exception:
                pass
            
        # 임시 파일 정리
        try:
            time.sleep(2)  # 파일 사용이 완전히 끝날 때까지 대기
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_error:
            st.warning(f"임시 파일 정리 중 오류 발생: {cleanup_error}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        st.error("자세한 오류 정보:", exc_info=True)
