import datetime
import os, sys
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import ChatMessage

from utils import *

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
        "load_report_llm": None,
        "load_RAG_retriever_llm": None,
        "load_RAG_generator_llm": None,
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

        st.write("## 초기화면")
        reset_btn = st.button("Reset")
        if reset_btn:
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()

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
    # documents_path = os.path.join(os.path.dirname(__file__), "..", "qa_llm", "dataset", "MinerU", "MyOCR", "results", "mineru_output")
    documents_path = os.path.join(os.path.dirname(__file__), "..", "qa_llm", "dataset", "MinerU", "MyOCR", "results_2", "mineru_output")

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
        
        if not st.session_state.load_report_llm:
            load_report_llm()
            st.experimental_rerun()
        
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
                
                # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # filename = f"report_result_{timestamp}.md"
                # with open(filename, "w", encoding="utf-8") as f:
                #     f.write(st.session_state.report)

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

if __name__ == '__main__':
    main()