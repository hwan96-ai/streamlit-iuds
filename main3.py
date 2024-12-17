__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import boto3  

st.set_page_config(
    page_title="상품 문의 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import sys
import os
from datetime import datetime
from langchain.schema.runnable import RunnablePassthrough
def get_bedrock_client():
    try:
        # Streamlit Cloud의 secrets에서 가져오기 시도
        aws_access_key_id = st.secrets.get("aws_access_key_id")
        aws_secret_access_key = st.secrets.get("aws_secret_access_key")
    except:
        # 로컬 환경변수에서 가져오기 시도
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS 인증 정보가 설정되지 않았습니다.")
    
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    return session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2"
    )
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
        
        # ChromaDB 설정 추가
        from chromadb.config import Settings
        import chromadb
        
        # ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(
            path=base_path,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            )
        )
        
        # Langchain Chroma 인스턴스 생성
        db = Chroma(
            client=client,
            embedding_function=embeddings,
        )
        
        return db
    except Exception as e:
        st.error(f"ChromaDB 로드 중 오류 발생: {str(e)}")
        raise

def get_unique_product_ids(db: Chroma):
    """Chroma DB에서 고유한 product_uuid 목록을 가져옵니다."""
    try:
        # 컬렉션의 모든 문서 가져오기
        collection = db._collection
        metadatas = collection.get()['metadatas']
        
        product_ids = set()
        for metadata in metadatas:
            if metadata and 'product_uuid' in metadata:
                product_ids.add(metadata['product_uuid'])
        
        return list(product_ids)
    except Exception as e:
        st.sidebar.error(f"제품 ID 조회 중 오류 발생: {str(e)}")
        return []
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
    
    def build_prompt(input_dict):
        current_time = get_current_datetime_with_day()
        docs = retrieve_docs(input_dict["question"], db, product_uuid)
        context = "\n".join([doc.page_content for doc in docs])
        
        messages = [
            ("system", f"""당신은 이 상품의 판매자입니다.
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
        {context}
        """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", input_dict["question"])
        ]
        
        return ChatPromptTemplate.from_messages(messages).format_messages(
            chat_history=input_dict.get("chat_history", []),
            question=input_dict["question"]
        )
    
    chain = RunnablePassthrough.assign() | build_prompt | llm | StrOutputParser()
    return chain
    
store = {}

def setup_conversation_with_history(chain):
    """채팅 히스토리를 설정하는 함수"""
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

def clear_chat_history():
    """채팅 관련 세션 상태를 초기화하는 함수"""
    keys_to_delete = ['messages', 'conversation_chain', 'current_product_id']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
def get_product_info_from_db(db: Chroma):
    """Chroma DB에서 제품 정보 가져오기"""
    try:
        # 모든 문서의 메타데이터 가져오기
        collection = db._collection
        metadatas = collection.get()['metadatas']
        
        # 제품 정보 매핑 생성 (product_uuid -> product_name)
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

        
# main 함수 정의
def main():
    # 메인 타이틀
    st.title("상품 문의 챗봇 🤖")
    
    # 현재 스크립트의 디렉토리 경로 가져오기
    db_path = "./chroma_db_1"
    
    try:
        # DB 로드
        db = load_chroma_db(db_path)
        
        # 제품 정보 가져오기
        product_info = get_product_info_from_db(db)
        
        if not product_info:
            st.error("제품 정보를 불러올 수 없습니다. 데이터베이스를 확인해주세요.")
            return
            
        # 사이드바 설정
        with st.sidebar:
            st.title("제품 선택 🛍️")
            st.markdown("---")
            
            # 제품 선택 드롭다운 (이름으로 표시, ID로 저장)
            product_names = {name: uuid for uuid, name in product_info.items()}
            selected_name = st.selectbox(
                "문의하실 제품을 선택하세요",
                options=list(product_names.keys()),
                key="product_selector"
            )
            selected_product_id = product_names[selected_name]
        
            st.markdown("---")
            
            # 채팅 기록 초기화 버튼
            if st.button("💫 채팅 기록 초기화", use_container_width=True):
                clear_chat_history()
                st.success("채팅 기록이 초기화되었습니다!")
                st.rerun()
            
            # 선택된 제품 정보 표시
            st.markdown("### 선택된 제품 정보")
            st.info(f"현재 선택: {selected_name}")
        
        # 세션 상태 초기화 또는 업데이트
        if 'conversation_chain' not in st.session_state or \
           'current_product_id' not in st.session_state or \
           st.session_state.current_product_id != selected_product_id:
            
            chain = create_rag_chain(db, selected_product_id)
            st.session_state.conversation_chain = setup_conversation_with_history(chain)
            st.session_state.current_product_id = selected_product_id
            
            # 제품이 변경되면 채팅 기록 초기화
            if 'messages' in st.session_state:
                clear_chat_history()
                st.rerun()
        
        # 채팅 히스토리 초기화
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # 시작 메시지 표시
        if not st.session_state.messages:
            st.markdown(f"""
            ### 👋 안녕하세요!
            **{product_info[selected_product_id]}**에 대해 궁금하신 점을 자유롭게 문의해주세요.
            """)
        
        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 채팅 입력
        if prompt := st.chat_input("무엇을 도와드릴까요?"):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 챗봇 응답 생성
            with st.chat_message("assistant"):
                with st.spinner('답변을 생성중입니다...'):
                    try:
                        # 응답 생성
                        chain_response = st.session_state.conversation_chain.invoke(
                            {"question": prompt},
                            config={"configurable": {"session_id": "unique_session_id"}}
                        )
                        
                        # reference_docs 가져오기
                        docs = retrieve_docs(prompt, db, selected_product_id)
                        
                        # 응답이 문자열인지 확인
                        if isinstance(chain_response, str):
                            response = chain_response
                        else:
                            response = chain_response.get('text', '')
                            
                        if not response:
                            raise ValueError("응답이 생성되지 않았습니다.")
                        
                        # 응답을 ### 구분자로 분리
                        answers = response.split("###")
                        
                        # 탭으로 다양한 응답 스타일 표시
                        tab1, tab2, tab3 = st.tabs(["💬 기본 답변", "✨ 창의적 답변", "😊 재미있는 답변"])
                        
                        with tab1:
                            st.markdown(answers[0].strip())
                            if docs and len(docs) > 0:
                                with st.expander("참고 문서 1"):
                                    st.markdown(f"**내용:**\n{docs[0].page_content}")
                                    st.markdown("**메타데이터:**")
                                    st.json(docs[0].metadata)
                        
                        with tab2:
                            st.markdown(answers[1].strip() if len(answers) > 1 else "답변이 없습니다.")
                            if docs and len(docs) > 1:
                                with st.expander("참고 문서 2"):
                                    st.markdown(f"**내용:**\n{docs[1].page_content}")
                                    st.markdown("**메타데이터:**")
                                    st.json(docs[1].metadata)
                        
                        with tab3:
                            st.markdown(answers[2].strip() if len(answers) > 2 else "답변이 없습니다.")
                            if docs and len(docs) > 2:
                                with st.expander("참고 문서 3"):
                                    st.markdown(f"**내용:**\n{docs[2].page_content}")
                                    st.markdown("**메타데이터:**")
                                    st.json(docs[2].metadata)
                        
                        # 첫 번째 답변을 채팅 히스토리에 저장
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answers[0].strip()}
                        )
                        
                    except Exception as e:
                        st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")
                        return
    
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.error(f"데이터베이스 경로: {db_path}")
        # 디버깅을 위한 추가 정보
        st.error("상세 오류 정보:", exc_info=True)
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        st.error("자세한 오류 정보")  # exc_info 파라미터 제거
