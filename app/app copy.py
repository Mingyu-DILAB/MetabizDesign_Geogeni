import os, sys
import tiktoken
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader

from utils import *

# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def basic_setting():
    st.set_page_config(
        page_title="Geogeni",
        page_icon=":books:",
        layout="wide"
    )
    st.title("Geogeni")
    
    # Checking if keys exist in session_state, if not, initializing them.
    defaults = {
        "report": None,
        "lat_lng_coordinates": [None, None], # user input
        "data_coordinates": [None, None], # coordinate system of data
        "screen": "map", # initial screen
        "store": dict(), # A dictionary to store conversation history of session
        "messages": [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar():
    global retrieval_distance, max_retrieval, retrieval_option
    
    with st.sidebar:
        st.write("## 시추공 검색")
        retrieval_distance = st.slider("Retrieval Distance (m)", 0, 1000, 500, help="지정한 좌표로부터 검색할 거리")
        max_retrieval = st.slider("Max Retrieval", 0, 20, 5, help="검색할 시추공 최대 개수")
        retrieval_option = st.radio(
            "Option",
            ["Primary", "Nearest"],
            help=f"Primary: Max Retrieval개가 검색되면 중단, Nearest: 모두 검색 후 가장 가까운 Max Retrieval개 필터링",
        )
        
        # st.write("## 기타")
        # uploaded_files = st.file_uploader("Upload your file", type=['pdf','docx'], accept_multiple_files=True)
        # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        # process = st.button("Process")
        
        # session_id = st.text_input("Session ID", value="ex")
        # clear_btn = st.button("Clear")
        # if clear_btn:
        #     st.session_state["messages"] = []
        #     st.session_state["store"] = dict()
        #     st.experimental_rerun()

        st.write("## 초기화면")
        reset_btn = st.button("Reset")
        if reset_btn:
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
    
    # if process:
    #     if not openai_api_key:
    #         st.info("Please add your OpenAI API key to continue.")
    #         st.stop()
    #     files_text = get_text(uploaded_files)
    #     text_chunks = get_text_chunks(files_text)
    #     vetorestore = get_vectorstore(text_chunks)
    #     st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
    #     st.session_state.processComplete = True

def get_coordinates():
    lat_lng_coordinates_container = st.empty()
    with lat_lng_coordinates_container:
        col1, col2 = st.columns(2)
        with col1:
            lat_input = st.text_input("Latitude")
        with col2:
            lng_input = st.text_input("Longitude")

    if st.button("Create Report"):
        st.session_state.lat_lng_coordinates = [float(lat_input), float(lng_input)]
        st.session_state.report = create_report(retrieval_distance, max_retrieval, retrieval_option)
        # try:
        #     st.session_state.lat_lng_coordinates = [float(lat_input), float(lng_input)]
        #     st.session_state.report = create_report(retrieval_distance, max_retrieval, retrieval_option)
        # except:
        #     st.error("Please enter valid numeric values for both latitude and longitude.")

def switch_screen(screen_name):
    st.session_state["screen"] = screen_name

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--retriever_model_name',
        choices=["text-embedding-3-small", "text-embedding-3-large", "BAAI/bge-m3"],
        default="BAAI/bge-m3",
    )
    p.add_argument(
        '--generator_model_name', 
        choices=["MLP-KTLim/llama-3-Korean-Bllossom-8B"],
        default="MLP-KTLim/llama-3-Korean-Bllossom-8B",
    )
    args = p.parse_args()
    
    return args

def load_chunks():
    documents_path = os.path.join(os.path.dirname(__file__), "..", "qa_llm", "dataset", "MinerU", "MyOCR", "results", "mineru_output")

    chunks = []
    for root, dirs, files in os.walk(documents_path):
        if "chunks.txt" in files:
            with open(os.path.join(root, "chunks.txt"), "r", encoding="utf-8") as chunk_file:
                lines = chunk_file.readlines()
                for line in lines:
                    chunks.append(line.strip())
            continue

    return chunks

def main():
    basic_setting()
    sidebar()

    # 초기 화면
    if st.session_state["screen"] == "map":
        load_map()
        get_coordinates()
        
        if st.session_state.report:
            switch_screen("report_and_qa")
            st.experimental_rerun()

    # 보고서 & QA 화면
    if st.session_state["screen"] == "report_and_qa":
        col1, col2 = st.columns([2, 3])

        with col1:
            container = st.container(border=True, height=900)
            with container:
                st.write(st.session_state.report)

        with col2:
            container = st.container(border=True, height=900)

            # Show previous chat
            for message in st.session_state.messages:
                container.chat_message(message.role).write(message.content)

            # Load chunks
            chunks = load_chunks()

            # Initialize the chatbot
            args = define_argparser()
            chatbot = RAGChatbot(
                chunks=chunks,
                retriever_model_name=args.retriever_model_name,
                generator_model_name=args.generator_model_name,
            )

            # Build FAISS index
            chatbot.build_index()

            # Enter messages from the user
            if user_input := st.chat_input("Geogeni 메시지 입력"):
                st.session_state.messages.append(ChatMessage(role="user", content=user_input))

                container.chat_message("user").write(user_input)

                with container.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        stream_handler = StreamlitHandler(st.empty())

                        response = chatbot.generate_response(user_input, st.session_state.report, top_k=5)

                        st.session_state["messages"].append(ChatMessage(role="assistant", content=response))
                        st.experimental_rerun()

        # with col2:
        #     container = st.container(border=True, height=900)

        #     # Show previous chat
        #     for message in st.session_state.messages:
        #         container.chat_message(message.role).write(message.content)
            
        #     # Enter messages from the user
        #     if user_input := st.chat_input("Geogeni 메시지 입력"):
        #         st.session_state.messages.append(ChatMessage(role="user", content=user_input))
                    
        #         container.chat_message("user").write(user_input)
                        
        #         with container.chat_message("assistant"):
        #             with st.spinner("Thinking..."):
        #                 stream_handler = StreamlitHandler(st.empty())
                    
        #                 prompt = basic_prompt()
        #                 llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        #                 chain = prompt | llm
                        
        #                 def get_session_history(session_id: str) -> BaseChatMessageHistory:
        #                     '''Retrieval of session records based on session ID'''
                            
        #                     if session_id not in st.session_state["store"]:
        #                         st.session_state["store"][session_id] = ChatMessageHistory()
        #                     return st.session_state["store"][session_id]
                        
        #                 chain_with_memory = (
        #                     RunnableWithMessageHistory(
        #                         chain,
        #                         get_session_history,
        #                         input_messages_key="question", # 체인을 invoke할 때 사용자 쿼리 입력으로 지정하는 key
        #                         history_messages_key="history", # 대화 기록으로 지정하는 key
        #                     )
        #                 )
                        
        #                 response = chain_with_memory.invoke(
        #                     {"question": user_input},
        #                     config={"configurable": {"session_id": "ex"}},
        #                 )
                        
        #                 st.session_state["messages"].append(ChatMessage(role="assistant", content=response.content))

if __name__ == '__main__':
    main()