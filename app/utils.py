import os, sys, json, math, random, argparse
import torch
import tiktoken
import logging
import pyproj
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
from flask import Blueprint, request, Flask
from collections import OrderedDict
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# A100 3번 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
seed(42)

ALL_TEST = {
    "토질시험": ["기본물성_기본물성시험", "토사_입도분석", "토사_일축압축", "토사_삼축압축_CU", "토사_삼축압축_UU", "토사_압밀시험", "토사_CBR"],
    "현장투수 및 수압시험": ["기본현장_현장수압시험", "기본현장_현장투수시험"],
    "표준관입시험": ["기본현장_표준관입시험"],
    "암석시험": ["암석_삼축압축", "암석_일축압축", "암석_절리면전단", "암석_점하중"],
    "하향식탄성파": ["물리검층_하향식탄성파"]
}

class Chunking_1_2():
    def __init__(self, file_path, extension="pdf"):
        self.file_path = file_path
        self.extension = extension
        
    def convert_tm_to_latlng(self, x_tm, y_tm, coord_system):
        coord_systems = {
            "(Bessel 경위도 )": "epsg:4004",
            "(GRS80 경위도 )": "epsg:4019",
            "WGS84": "epsg:4326",
            "(WGS84 경위도 )": "epsg:4326",
            "(GRS80 UTMK )": "epsg:5179",
            "(Bessel TM 서부원점(20만,50만))": "epsg:5173",
            "(Bessel TM 중부원점(20만,50만))": "epsg:5174",
            "(Bessel TM 동부원점(20만,50만))": "epsg:5176",
            "(Bessel TM 동해원점(20만,50만))": "epsg:5177",
            "(GRS80 TM 서부원점(20만,50만))": "epsg:5180",
            "(GRS80 TM 서부원점(20만,60만))": "epsg:5185",
            "(GRS80 TM 중부원점(20만,50만))": "epsg:5181",
            "(GRS80 TM 중부원점(20만,60만))": "epsg:5186",
            "(GRS80 TM 동부원점(20만,50만))": "epsg:5183",
            "(GRS80 TM 동부원점(20만,60만))": "epsg:5187",
            "(GRS80 TM 동해원점(20만,50만))": "epsg:5184",
            "(GRS80 TM 동해원점(20만,60만))": "epsg:5188",
            "(WGS84 UTM 51N )": "epsg:32651",
            "(WGS84 UTM 52N )": "epsg:32652",
        }
        
        wgs84 = pyproj.CRS(coord_systems["WGS84"])  # 원래 좌표계
        tm_korea = pyproj.CRS(coord_systems[coord_system])  # 변환하려는 좌표계
        
        transformer = pyproj.Transformer.from_crs(tm_korea, wgs84)

        lat, lng = transformer.transform(y_tm, x_tm)
        
        return lat, lng

    def get_address(self, lat, lng):
        url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
        api_key = "6e4e709ab806a5cea76e60844e9d612a"
        headers = {"Authorization": f"KakaoAK {api_key}"}
        params = {"x": lng, "y": lat}
        response = requests.get(url, headers=headers, params=params)

        try:
            address_name = response.json()["documents"][0]["address"]["address_name"]
            return address_name
        # 카카오 api로 주소가 반환이 되지 않는 경우
        except:
            geolocator = Nominatim(user_agent="Geogeni", timeout=None)
            location = geolocator.reverse((lat, lng), exactly_one=True)
            return location.address if location else "주소를 찾을 수 없습니다."
        
    def _pdf_load(self, table_extraction="camelot"):
        pdf_docs = {"texts": [], "tables": []}
        table_strings = []
        
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]  # 파일 이름 추출
        current_directory_name = os.path.basename(os.path.dirname(self.file_path))  # 최하위 폴더명 추출

        # tables
        if table_extraction == "camelot":
            tables = camelot.read_pdf(self.file_path, pages="all")
            for i in range(len(tables)):
                table = tables[i].df
                pdf_docs["tables"].append(table)    # pdf2pdf -> return pdf_docs["tables"]
                
                # markdown_string = table.to_markdown(index=False)  # pdf2markdown
                csv_string = table.to_csv(index=False, header=False)    # pdf2csv
                table_strings.append(csv_string)
                
        return table_strings
    
    def _xlsx_load(self) -> str:
        df = pd.read_excel(self.file_path)
        
        if os.path.split(self.file_path)[-1].split("_")[1] == "표준관입시험":
            new_column_values = []
            
            def custom_round(value):
                '''round의 사사오입 규칙 제거'''
                if value - int(value) == 0.5:
                    return math.ceil(value)
                else:
                    return round(value)
                
            depths = df["기본현장_표준관입_심도(-m)"].dropna()
            for depth in depths:
                rounded_depth = custom_round(depth)
                depth_list = df["기본현장_표준관입_표준관입심도(-m)"].dropna()
                try:
                    idx = depth_list[depth_list == rounded_depth].index[0]
                    new_column_values.append(df["기본현장_표준관입_표준관입시험"].iloc[idx])
                except IndexError:
                    pass
                
            df = df.drop("기본현장_표준관입_표준관입심도(-m)", axis=1)
            df = df.drop("기본현장_표준관입_표준관입시험", axis=1)
            df["기본현장_표준관입_표준관입시험"] = new_column_values + [np.nan] * (len(df) - len(new_column_values))
            
        xlsx_docs = df.dropna(how="all", axis=0).to_csv(index=False)
        return xlsx_docs
    
    def _json_load(self, sich_path) -> str:
        pj = os.path.split(os.path.split(sich_path)[0])[-1]
        sich = os.path.split(sich_path)[-1]
        
        # data의 위도, 경도 수정(or 추가)
        with open(os.path.join(os.path.dirname(__file__), "..", "dataset", "pjCoord2.json"), "r", encoding="utf-8") as pjCoord_file:
            pjCoord = json.load(pjCoord_file)
            x_tm = float(pjCoord[pj][sich]["X_Coord"].split(":")[-1])
            y_tm = float(pjCoord[pj][sich]["Y_Coord"].split(":")[-1])
            coord_system = pjCoord[pj][sich]["좌표계"]
            
            # 좌표계 변환
            lat, lng = self.convert_tm_to_latlng(x_tm, y_tm, coord_system)
            address = self.get_address(round(lat, 8), round(lng, 8))
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data["LL"] = {"X": round(lat, 8), "Y": round(lng, 8)}
            data["address"] = address

        # 생각해보니 LL, address만 필요해서 우선 얘내만 활용
        json_docs = {k: data[k] for k in ["LL", "address"] if k in data}

        # json_docs = json.dumps(data, ensure_ascii=False)
        return json_docs
    
    def get_docs(self, sich_path=None):
        if self.extension == "pdf":
            pdf_docs = self._pdf_load()
            return pdf_docs
        
        elif self.extension == "xlsx":
            xlsx_docs = self._xlsx_load()
            return xlsx_docs
        
        elif self.extension == "json":
            json_docs = self._json_load(sich_path)
            return json_docs

class Chunking_3():
    def __init__(self, file_path, extension="pdf"):
        self.file_path = file_path
        self.extension = extension

    def data_preprocess(self, data):
        if self.extension == "pdf":
            pass

        elif self.extension == "xlsx":
            data = data.to_dict("list")

            # 뒤에 붙은 nan 값들 제거 (공통)
            def remove_trailing_nans(value_list):
                for i in range(len(value_list) -1, -1, -1):
                    value = value_list[i]
                    if not (pd.isna(value) or (isinstance(value, str) and value.lower() == 'nan')):
                        return value_list[:i+1]
                return []

            for key in data:
                data[key] = remove_trailing_nans(data[key])

            # 시험 별 전처리
            def process_for_each_test(data, key_available_list):
                for key in list(data.keys()):
                    remove_key = True

                    for key_available in key_available_list:
                        if (key.strip() in key_available) or (key_available in key.strip()):
                            remove_key = False

                    if remove_key:
                        del data[key]

                return data

            test_name = os.path.split(os.path.split(self.file_path)[0])[-1]

            if test_name == "기본물성_기본물성시험":
                key_available_list = ["기본물성_기본물성_심도", "기본물성_기본물성_함수율", "기본물성_기본물성_비중", "기본물성_기본물성_액성한계", "기본물성_기본물성_소성지수", "기본물성_기본물성_USCS"]
                data = process_for_each_test(data, key_available_list)

            elif test_name == "토사_입도분석":
                key_available_list = ["토사_입도분석_심도", "토사_입도분석_체통과백분율#4", "토사_입도분석_체통과백분율#10", "토사_입도분석_체통과백분율#40", "토사_입도분석_체통과백분율#200", "토사_입도분석_체통과백분율#0.005mm이하"]
                data = process_for_each_test(data, key_available_list)

                for key in list(data.copy().keys()):
                    if "체통과백분율#100" in key.strip():
                        del data[key]

            elif test_name == "토사_일축압축":
                key_available_list = ["토사_일축압축_심도", "토사_일축압축_자연시료압축강도"]
                data = process_for_each_test(data, key_available_list)

            elif test_name == "토사_삼축압축_CU":
                key_available_list = ["토사_삼축압축CU_심도", "토사_삼축압축CU_점착력", "토사_삼축압축CU_내부마찰각"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "토사_삼축압축_UU":
                key_available_list = ["토사_삼축압축UU_심도", "토사_삼축압축UU_점착력"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "토사_압밀시험":
                key_available_list = ["토사_압밀_심도", "토사_압밀_선행압밀하중", "토사_압밀_압축지수"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "토사_CBR":
                key_available_list = ["토사_CBR_심도", "A다짐", "B다짐", "D다짐"]
                data = process_for_each_test(data, key_available_list)

                for key in list(data.copy().keys()):
                    if ("B다짐" in key) and ("건조밀도" in key):
                        data["토사_CBR_D다짐최대건조밀도 (kN/m3)"] = data[key]
                        del data[key]

                    if ("B다짐" in key) and ("최적함수비" in key):
                        data["토사_CBR_D다짐최적함수비(OMC, %)"] = data[key]
                        del data[key]

            elif test_name == "기본현장_현장투수시험":
                key_available_list = ["기본현장_현장투수_심도", "기본현장_현장투수_평균투수계수"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "기본현장_현장수압시험":
                key_available_list = ["기본현장_현장수압_심도", "기본현장_현장수압_시간간격", "기본현장_현장수압_수압", "기본현장_현장수압_평균투수계수", "기본현장_현장수압_평균루전값"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "기본현장_표준관입시험":
                key_available_list = ["기본현장_표준관입_표준관입심도", "기본현장_표준관입_표준관입시험"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "암석_일축압축":
                key_available_list = ["암석_일축압축_심도", "암석_일축압축_일축압축강도", "암석_일축압축_탄성계수"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "암석_삼축압축":
                key_available_list = ["암석_삼축압축_심도", "암석_삼축압축_점착력", "암석_삼축압축_내부마찰각"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "암석_점하중":
                key_available_list = ["암석_점하중_심도", "암석_점하중_점하중강도", "암석_점하중_일축압축강도"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "암석_절리면전단":
                key_available_list = ["암석_절리면전단_심도", "암석_절리면전단_점착력", "암석_절리면전단_절리면압축강도", "암석_절리면전단_내부마찰각", "암석_절리면전단_수직응력", "암석_절리면전단_전단응력"]
                data = process_for_each_test(data, key_available_list)
                
            elif test_name == "물리검층_하향식탄성파":
                key_available_list = ["물리검층_하향식탄성파_심도", "물리검층_하향식탄성파_전단파속도P파", "물리검층_하향식탄성파_전단파속도S파", "물리검층_하향식탄성파_포아송비", "물리검층_하향식탄성파_전단탄성계수", "물리검층_하향식탄성파_영률", "물리검층_하향식탄성파_밀도"]
                data = process_for_each_test(data, key_available_list)

                for key in list(data.copy().keys()):
                    for key_available in key_available_list:
                        if (key.strip() in key_available) or (key_available in key.strip()):
                            data[key_available] = data[key]
                            del data[key]
            
            for key in list(data.copy().keys()):
                if ("\n" in key):
                    new_key = key.replace("\n", "").strip()

                    data[new_key] = data[key]
                    del data[key]

            return data

        elif self.extension == "json":
            pass
        
    def _pdf_load(self, table_extraction="camelot"):
        pdf_docs = {"texts": [], "tables": []}
        table_strings = []
        
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]  # 파일 이름 추출
        current_directory_name = os.path.basename(os.path.dirname(self.file_path))  # 최하위 폴더명 추출

        # # tables
        # if table_extraction == "camelot":
        #     tables = camelot.read_pdf(self.file_path, pages="all")
        #     for i in range(len(tables)):
        #         table = tables[i].df
        #         pdf_docs["tables"].append(table)    # pdf2pdf -> return pdf_docs["tables"]
                
        #         # markdown_string = table.to_markdown(index=False)  # pdf2markdown
        #         csv_string = table.to_csv(index=False, header=False)    # pdf2csv
        #         table_strings.append(csv_string)
                
        # return table_strings
    
    def _xlsx_load(self):
        try:
            data = pd.read_excel(self.file_path, sheet_name=None)
            values = list(data.values())
            data = pd.concat(values, ignore_index=True)

        except:
            return pd.DataFrame()

        xlsx_docs = self.data_preprocess(data)
        return xlsx_docs
    
    def _json_load(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        json_docs = json.dumps(data, ensure_ascii=False)
        return json_docs
    
    def get_docs(self):
        if self.extension == "pdf":
            pdf_docs = self._pdf_load()
            return pdf_docs
        
        elif self.extension == "xlsx":
            xlsx_docs = self._xlsx_load()
            return xlsx_docs
        
        elif self.extension == "json":
            json_docs = self._json_load()
            return json_docs

class StreamlitHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def load_map():
    map_container = st.empty()

    with map_container:
        # 지도 파일 로드
        html_file_path = os.path.join(os.path.dirname(__file__), "custom_map", "map.html")
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

            # HTML 삽입 (iframe 내부에서 좌표 전송하는 스크립트 추가)
            components.html(html_content + '''
                <script>
                    // 부모 프레임에서 메시지를 수신하는 이벤트 리스너
                    window.addEventListener("message", (event) => {
                        if (event.data.type === 'coordinates') {
                            console.log("Coordinates received by iframe:", event.data.lat, event.data.lng);
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: { lat: event.data.lat, lng: event.data.lng }
                            }, "*");
                        }
                    });
                </script>
            ''', height=800, scrolling=True)


def haversine(criterion_lat, criterion_lng, lat, lng):
    '''Haversine 공식에 따른 두 지점 간의 거리 계산 함수'''

    R = 6371000 # 지구 반지름 (m)
    
    # 위도와 경도를 라디안으로 변환
    phi1 = math.radians(criterion_lat)
    phi2 = math.radians(lat)
    delta_phi = math.radians(lat - criterion_lat)
    delta_lambda = math.radians(lng - criterion_lng)
    
    # Haversine 공식 계산
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # 두 지점 간의 거리
    distance = R * c
    
    return distance

def convert_latlng_to_tm(lat, lng):
    wgs84 = pyproj.CRS('epsg:4326')  # 원래 좌표계 (WGS84)
    # tm_korea = pyproj.CRS('epsg:5179')  # 변환하려는 좌표계 (UTM-K)
    tm_korea = pyproj.CRS('epsg:5186')  # 변환하려는 좌표계 (TM-중부원점) (20만, 60만)
    
    # Transformer 생성
    transformer = pyproj.Transformer.from_crs(wgs84, tm_korea)
    
    # 좌표 변환
    x_tm, y_tm = transformer.transform(lat, lng)  # 주의: transform 함수는 lat 먼저 받습니다.
    
    return x_tm, y_tm

def convert_tm_to_latlng(x_tm, y_tm, coord_system):
    coord_systems = {
        "(Bessel 경위도 )": "epsg:4004",
        "(GRS80 경위도 )": "epsg:4019",
        "WGS84": "epsg:4326",
        "(WGS84 경위도 )": "epsg:4326",
        "(GRS80 UTMK )": "epsg:5179",
        "(Bessel TM 서부원점(20만,50만))": "epsg:5173",
        "(Bessel TM 중부원점(20만,50만))": "epsg:5174",
        "(Bessel TM 동부원점(20만,50만))": "epsg:5176",
        "(Bessel TM 동해원점(20만,50만))": "epsg:5177",
        "(GRS80 TM 서부원점(20만,50만))": "epsg:5180",
        "(GRS80 TM 서부원점(20만,60만))": "epsg:5185",
        "(GRS80 TM 중부원점(20만,50만))": "epsg:5181",
        "(GRS80 TM 중부원점(20만,60만))": "epsg:5186",
        "(GRS80 TM 동부원점(20만,50만))": "epsg:5183",
        "(GRS80 TM 동부원점(20만,60만))": "epsg:5187",
        "(GRS80 TM 동해원점(20만,50만))": "epsg:5184",
        "(GRS80 TM 동해원점(20만,60만))": "epsg:5188",
        "(WGS84 UTM 51N )": "epsg:32651",
        "(WGS84 UTM 52N )": "epsg:32652",
    }
    
    wgs84 = pyproj.CRS(coord_systems["WGS84"])  # 원래 좌표계
    tm_korea = pyproj.CRS(coord_systems[coord_system])  # 변환하려는 좌표계
    
    transformer = pyproj.Transformer.from_crs(tm_korea, wgs84)

    lat, lng = transformer.transform(y_tm, x_tm)
    
    return lat, lng

def find_sichs_within_radius(retrieval_distance: int = 500, max_retrieval: int = 5, retrieval_option: str = "Primary"):
    print("Find sichs within radius...")
    criterion_lat, criterion_lng = st.session_state.lat_lng_coordinates
    retrieval_distance = 50000
    
    # retrieval_distance 내에 있는 시추공 데이터 검색
    retrieved_sichs = dict()
    with open(os.path.join(os.path.dirname(__file__), "..", "dataset", "pjCoord2.json"), 'r', encoding="utf-8") as f:
        pjCoord = json.load(f)
        
        if retrieval_option == "Primary":
            primary_retrieval_filled = False
            
            for pj_k, pj_v in tqdm(pjCoord.items()):
                if primary_retrieval_filled: # 최대 시추공 개수만큼 검색되면, 검색 종료
                    break
                
                for sich_k, sich_v in pj_v.items():
                    try:
                        x_tm = float(sich_v["X_Coord"].split(":")[-1])
                        y_tm = float(sich_v["Y_Coord"].split(":")[-1])
                        coord_system = sich_v["좌표계"]
                        
                        # 좌표계 변환
                        lat, lng = convert_tm_to_latlng(x_tm, y_tm, coord_system)
                        
                        # haversine 거리 계산
                        distance = haversine(criterion_lat, criterion_lng, lat, lng)
                        if distance < retrieval_distance:
                            retrieved_sichs[sich_k] = sich_v
                            
                            if len(retrieved_sichs) == max_retrieval:
                                primary_retrieval_filled = True
                                break
                            
                    except:
                        pass
        
        elif retrieval_option == "Nearest":
            furthest_sich_of_nearest = ""
            furthest_dist_of_nearest = float("-inf")
            
            for pj_k, pj_v in tqdm(pjCoord.items()):
                for sich_k, sich_v in pj_v.items():
                    try:
                        x_tm = float(sich_v["X_Coord"].split(":")[-1])
                        y_tm = float(sich_v["Y_Coord"].split(":")[-1])
                        coord_system = sich_v["좌표계"]
                        
                        # 좌표계 변환
                        lat, lng = convert_tm_to_latlng(x_tm, y_tm, coord_system)
                        
                        # haversine 거리 계산
                        distance = haversine(criterion_lat, criterion_lng, lat, lng)
                        if distance < retrieval_distance:
                            if len(retrieved_sichs) < max_retrieval:
                                if distance > furthest_dist_of_nearest:
                                    furthest_sich_of_nearest = sich_k
                                    furthest_dist_of_nearest = distance
                                retrieved_sichs[sich_k] = sich_v
                            
                            elif len(retrieved_sichs) == max_retrieval:
                                if distance >= furthest_dist_of_nearest:
                                    continue
                                
                                else:
                                    del retrieved_sichs[furthest_sich_of_nearest]
                                    retrieved_sichs[sich_k] = sich_v
                                    
                                    furthest_dist_of_nearest = float("-inf")
                                    for retrieved_sich_k, retrieved_sich_v in retrieved_sichs.items():
                                        x_tm = float(sich_v["X_coord"])
                                        y_tm = float(sich_v["Y_coord"])
                                        coord_system = sich_v["좌표계"]
                                        lat, lng = convert_tm_to_latlng(x_tm, y_tm, coord_system)
                                        
                                        distance = haversine(criterion_lat, criterion_lng, lat, lng)
                                        if distance > furthest_dist_of_nearest:
                                            furthest_sich_of_nearest = retrieved_sich_k
                                            furthest_dist_of_nearest = distance
                    
                    except:
                        pass
    
    return retrieved_sichs

def preprocess_1_2(retrieved_sich_paths):
    '''1_2'''
    text = """"""

    # 시험 개수 직접 카운트(2-2 시험 DB 내역에 활용)
    n_test = {"기본물성_기본물성시험": 0, "기본현장_표준관입시험": 0, "기본현장_현장수압시험": 0, "기본현장_현장투수시험": 0, "물리검층_하향식탄성파": 0, "암석_일축압축": 0, "토사_입도분석": 0}
    for i, (sich_name, retrieved_sich_path) in enumerate(retrieved_sich_paths.items()):

        for test in os.listdir(retrieved_sich_path):
            test_path = os.path.join(retrieved_sich_path, test)
            if os.path.isdir(test_path):
                try:
                    n_test[test] += 1
                except:
                    pass

    n_test_category = {"현장시험": {"표준관입시험": n_test["기본현장_표준관입시험"], "현장수압시험": n_test["기본현장_현장수압시험"], "현장투수시험": n_test["기본현장_현장투수시험"]}, "기본물성시험": {"기본물성시험": n_test["기본물성_기본물성시험"]}, "토사시험": {"입도분석": n_test["토사_입도분석"]}, "암석시험": {"일축압축시험": n_test["암석_일축압축"]}, "물리검층": {"하향식탄성파": n_test["물리검층_하향식탄성파"]}}
    for main_category in n_test_category.copy():
        n_test_category[main_category] = {k: v for k, v in n_test_category[main_category].items() if v != 0}
        if not n_test_category[main_category]:
            del n_test_category[main_category]

    # 지층 분석(2-3 지층 개요에 활용)
    stratum = {}
    for i, (sich_name, retrieved_sich_path) in enumerate(retrieved_sich_paths.items()):
        try:
            with open(os.path.join(retrieved_sich_path, "기본현장_표준관입시험", "기본현장_표준관입시험" + f"_{sich_name}.json"), "r") as f:
                layers = json.load(f)["LAYER"]

                for layer in layers:
                    if layer["name"] not in stratum:
                        stratum[layer["name"]] = {
                            "층후": {"from": layer["from"], "to": layer["to"]},
                            "desc": [f"시추공 {sich_name}: {layer['desc']}"]
                        }
                    else:
                        if layer["from"] < stratum[layer["name"]]["층후"]["from"]:
                            stratum[layer["name"]]["층후"]["from"] = layer["from"]
                        
                        if layer["to"] > stratum[layer["name"]]["층후"]["to"]:
                            stratum[layer["name"]]["층후"]["to"] = layer["to"]

                        stratum[layer["name"]]["desc"].append(f"시추공 {sich_name}: {layer['desc']}")

        # 기본현장_표준관입시험 폴더가 없는 경우
        except:
            pass

    def get_layer_type_priority(key):
        return 0 if "토층" in key else 1

    try:
        stratum = dict(sorted(stratum.items(), key=lambda x: (get_layer_type_priority(x[0]), x[1]['층후']['from'], x[1]['층후']['to'])))
    except:
        pass

    # Context
    context = {"시험 DB 내역": n_test_category, "지층 개요": stratum}
    context = json.dumps(context, ensure_ascii=False)
    context = context.replace("{", "[").replace("}", "]")
    text += f"""# Context\n{context}\n\n"""

    # Each sich context
    each_sich_context = {}
    for i, (sich_name, retrieved_sich_path) in enumerate(retrieved_sich_paths.items()):
        # sich_meta
        sich_meta_path = os.path.join(retrieved_sich_path, sich_name + "_sich_meta.json")
        if os.path.exists(sich_meta_path):
            with open(sich_meta_path, "r", encoding="utf-8") as f:
                sich_meta = json.load(f)

            each_sich_context = {k: v for k, v in sich_meta.items() if k in ["조사명", "구분", "기간", "시추공명", "시추공코드", "심도", "표고", "지하수위"]}

        # coordinates, address
        for test in os.listdir(retrieved_sich_path):
            test_path = os.path.join(retrieved_sich_path, test)
            if os.path.isdir(test_path):
                for root_test, _, test_files in os.walk(test_path):
                    for file in sorted(test_files):
                        file_path = os.path.join(root_test, file)
                        if os.path.isfile(file_path):
                            try:
                                extension = file.split(".")[-1]
                                if extension == "pdf":
                                    continue
                                # 엑셀도 우선 필요없어 보여서 패스
                                elif extension == "xlsx":
                                    continue
                                    # chunk = Chunking_1_2(file_path=file_path, extension=extension).get_docs()
                                elif extension == "json":
                                    if test == "기본현장_표준관입시험":
                                        chunk = Chunking_1_2(file_path=file_path, extension=extension).get_docs(retrieved_sich_path)
                                        for k, v in chunk.items():
                                            each_sich_context[k] = chunk[k]
                                    
                            except Exception as ex:
                                print(f"Error processing file {file_path}: {ex}")
                        else:
                            print(f"Skipping directory (not a file): {file_path}")
        
        each_sich_context = json.dumps(each_sich_context, ensure_ascii=False)
        each_sich_context = each_sich_context.replace("{", "[").replace("}", "]")

        text += f"""## 시추공 {i+1}: {sich_name}\n{each_sich_context}\n\n"""
    return text

def preprocess_3(retrieved_sich_paths):
    text = """"""

    # all_test 변수에 모든 시험 정보들 저장
    all_test = {"토질시험": {sich_name: {} for sich_name in retrieved_sich_paths.keys()}, "현장투수 및 수압시험": {sich_name: {} for sich_name in retrieved_sich_paths.keys()}, "표준관입시험": {sich_name: {} for sich_name in retrieved_sich_paths.keys()}, "암석시험": {sich_name: {} for sich_name in retrieved_sich_paths.keys()}, "하향식탄성파": {sich_name: {} for sich_name in retrieved_sich_paths.keys()}}
    for i, (sich_name, retrieved_sich_path) in enumerate(retrieved_sich_paths.items()):                    
        # 시험 레벨
        for test_name in os.listdir(retrieved_sich_path):
            test_path = os.path.join(retrieved_sich_path, test_name)

            if os.path.isdir(test_path):

                # 폴더명을 기준으로 메인 카테고리 분류
                if test_name in ["기본물성_기본물성시험", "토사_입도분석", "토사_일축압축", "토사_CBR", "토사_삼축압축_CU", "토사_삼축압축_UU", "토사_압밀시험"]:
                    main_category = "토질시험"
                elif test_name in ["기본현장_현장수압시험", "기본현장_현장투수시험"]:
                    main_category = "현장투수 및 수압시험"
                elif test_name in ["기본현장_표준관입시험"]:
                    main_category = "표준관입시험"
                elif test_name in ["암석_삼축압축", "암석_일축압축", "암석_절리면전단", "암석_점하중"]:
                    main_category = "암석시험"
                elif test_name in ["물리검층_하향식탄성파"]:
                    main_category = "하향식탄성파"
                else:
                    main_category = None
                    continue

                # 청킹
                for file in os.listdir(test_path):
                    file_path = os.path.join(test_path, file)
                    if os.path.isfile(file_path):
                        extension = file.split(".")[-1]
                        if extension == "pdf":
                            continue
                        elif extension == "xlsx":
                            all_test[main_category][sich_name][file] = Chunking_3(file_path=file_path, extension=extension).get_docs()
                        elif extension == "json":
                            continue
                    else:
                        print(f"Skipping directory (not a file): {file_path}")

    # 시험들 중 정보가 하나도 없다면 제거
    for main_category, all_info in all_test.copy().items():
        is_info = False
        for sich_code, info in all_info.copy().items():
            if len(info):
                is_info = True
            else:
                del all_test[main_category][sich_code]

        if not is_info:
            del all_test[main_category]

    # 각 시추공마다 심도 열의 길이에 맞춰서 뒤에 nan 값 추가
    try:
        for main_category, all_info in all_test.copy().items():
            for sich_code, info in all_info.copy().items():
                for file_name, all_columns in info.copy().items():
                    max_length = max([len(value_list) for col_name, value_list in all_columns.items()])

                    for col_name, value_list in all_columns.items():
                        current_length = len(value_list)
                        if current_length < max_length:
                            padding_length = max_length - current_length
                            all_test[main_category][sich_code][file_name][col_name].extend([float('nan')] * padding_length)
    except:
        pass

    return all_test

def get_response_1_2(text_1_2):    
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    lora_adapter_path = os.path.join(os.path.dirname(__file__), "..", "report_llm", "llm", "report_1_2", "Qwen2.5-7B-Instruct_1epoch_1batch_53963")
    merged_model_path = os.path.join(os.path.dirname(__file__), "..", "report_llm", "llm", "report_1_2", "merged_model_2")

    prompt = """
    ### Instruction:
    당신은 주어진 context를 활용하여 지반조사 Report를 작성하는 데 특화된 지반 도메인 전문가입니다.
    지반조사 Report는 주어진 예시와 같이 1. 지역정보, 2. 조사내역 요약 순서로 반드시 작성되어야 하며, 결론 없이 마무리해야 합니다.
    1. 지역정보는 주어진 context에서 address 정보만 활용하며 사전지식에 기반하여 작성합니다. address 정보로부터 조사지역의 위치를 파악하고, 조사지역 위치로부터 1km 이내 부근에 대한 교통현황, 산계, 인근시설 등과 같은 요약 정보를 작성해야 합니다. 1km 이내의 범위에 있는 학교나 저수지, 아파트, 빌딩, 지하철역 등과 같은 큰 건물에 대해 작성하고, 만약 이러한 주변 정보가 없다면, 억지로 조사지역 인근에 대한 요약 정보를 작성하지 마세요.
    2. 조사내역 요약은 먼저 '1) 주변 시추공 좌표 정보'를 표로 작성합니다. 표의 열은 시추공코드, 위도, 경도, 지하수위(-m), 표고(m)로 구성되며, 위도와 경도는 각 시추공의 context에서 'LL'의 값을 참고하여 예시와 같이 작성하세요. 지하수위와 표고는 각 시추공의 context에서 '지하수위', '표고'를 참고하여 예시와 같이 작성하세요.
    그리고 '2) 시험 DB 내역 (실내시험, 현장시험 등)'을 표로 작성합니다. 주어진 context의 '시험 DB 내역'을 참조하여 예시와 같이 각 시험에 대한 시험명과 시험 입력 개수를 표로 작성하세요.
    마지막으로 '3) 지층 개요'를 표로 작성합니다. 주어진 context의 '지층 개요'를 참조하여 예시와 같이 지층 개요에 대한 표를 작성합니다. 표의 열은 '구분', '토질', '상대밀도', '층후(m)'로 구성되며, '토질' 열과 '상대밀도' 열은 context에서 각 지층의 'desc'를 참조하여 작성합니다. 특히, '상대밀도' 열의 값은 'desc'에 여러 시추공의 상대밀도 정보가 있을 수도 있고 전혀 없을 수도 있습니다. 만약 여러 시추공의 정보가 있다면 이들의 상대밀도들을 종합하여 최소~최대 상대밀도를 표에 작성하고, 상대밀도 정보를 찾을 수 없다면 억지로 추측해서 작성하지 말고 반드시 '-'로 표시하세요.
    """

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map='auto',
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        cache_dir=merged_model_path
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     merged_model_path,
    #     device_map='cuda',
    #     torch_dtype=torch.float16,
    #     quantization_config=quantization_config
    # )
    model.eval()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text_1_2}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_1_2 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response_1_2.strip()

def get_response_3(all_test):
    response_3 = """"""

    # 메인 카테고리 별로 보고서 생성
    for category_idx, (main_category, all_info) in enumerate(all_test.items()):
        print(f"Generating report ({category_idx+1}) {main_category}...", end=" ")

        user_input = """"""
        user_input += f"### 시험번호: {category_idx+1}\n"
        user_input += f"주 시험명: {main_category}\n"

        # 존재하는 하위 시험들 list-up
        sub_categories = ALL_TEST[main_category]
        sub_tests = {sub_category: [] for sub_category in sub_categories}

        for sich_code, info in all_info.items():
            file_names = list(info.keys())

            # 파일 이름들을 순회하면서, 어떤 시험에 속하는지 검색
            for file_name in file_names:
                for sub_category in sub_categories:
                    if file_name.startswith(sub_category):
                        sub_tests[sub_category].append(file_name)
                        break
        
        # 저장 포멧을 '시추공' 기준에서 '하위 시험'으로 변경
        new_all_info = {sub_category: {} for sub_category, sub_files in sub_tests.items()}

        for sub_category, sub_files in sub_tests.items():
            for sub_file in sub_files:

                for info in list(all_info.values()):
                    for file_name, content in info.items():
                        if file_name == sub_file:
                            new_all_info[sub_category][file_name] = content

        new_all_info = {k: v for k, v in new_all_info.items() if v}

        # 값이 아무것도 없다면 해당 시험은 스킵
        is_skip = True
        for sub_category, info in new_all_info.items():
            for file_name, content in info.items():
                for key, value in content.items():
                    if value:
                        is_skip = False
                        break
        if is_skip:
            print("Skip")
            continue

        # 시추공코드 내림차순 정렬
        sorted_new_all_info = dict()
        for sub_category, info in new_all_info.items():
            sorted_file_names = sorted(info.keys())

            sorted_dict = OrderedDict()
            for file_name in sorted_file_names:
                value = info[file_name]
                sorted_dict[file_name] = value

            sorted_new_all_info[sub_category] = sorted_dict

        sub_test_1 = {}
        sub_test_2 = {}
        sub_test_3 = {}
        for sub_category_idx, (sub_category, info) in enumerate(sorted_new_all_info.items()):
            if sub_category in ['기본물성_기본물성시험', '토사_입도분석', '기본현장_현장수압시험', '기본현장_현장투수시험', '기본현장_표준관입시험', '암석_삼축압축', '암석_일축압축', '암석_점하중', '물리검층_하향식탄성파']:
                for file_name, content in info.items():
                    sich_code = file_name.split(".")[0].split("_")[-1]

                    if sich_code not in sub_test_1.keys():
                        sub_test_1[sich_code] = {}
                    sub_test_1[sich_code][file_name] = content
                    
            elif sub_category in ['토사_일축압축', '토사_삼축압축_CU', '토사_삼축압축_UU', '토사_압밀시험', '암석_절리면전단']:
                for file_name, content in info.items():
                    sich_code = file_name.split(".")[0].split("_")[-1]

                    if sich_code not in sub_test_2.keys():
                        sub_test_2[sich_code] = {}
                    sub_test_2[sich_code][file_name] = content
                
            elif sub_category in ['토사_CBR']:
                for file_name, content in info.items():
                    sich_code = file_name.split(".")[0].split("_")[-1]

                    if sich_code not in sub_test_3.keys():
                        sub_test_3[sich_code] = {}
                    sub_test_3[sich_code][file_name] = content
        
        # 같은 표를 구성하는 시험 내에서, 어떤 시험은 존재하나 다른 시험은 존재하지 않을 경우
        # 존재하지 않는 시험에 대해 nan 값으로라도 추가하여 LLM이 아예 생성하지 않는 경우를 방지
        if main_category == "토질시험":
            if sub_test_1:
                for sich_code, info in sub_test_1.copy().items():
                    all_tests = [k.replace(k.split("_")[-1], "")[:-1] for k in info.keys()]
                    if ("기본물성_기본물성시험" in all_tests) and ("토사_입도분석" not in all_tests):
                        new_key = f"토사_입도분석_{sich_code}.xlsx"
                        sub_test_1[sich_code][new_key] = {"토사_입도분석_심도": [depth for depth in range(len(sub_test_1[sich_code][f"기본물성_기본물성시험_{sich_code}.xlsx"]["기본물성_기본물성_심도(G L, -m)"]))], "토사_입도분석_체통과백분율#4": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본물성_기본물성시험_{sich_code}.xlsx"]["기본물성_기본물성_심도(G L, -m)"]))], "토사_입도분석_체통과백분율#10": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본물성_기본물성시험_{sich_code}.xlsx"]["기본물성_기본물성_심도(G L, -m)"]))], "토사_입도분석_체통과백분율#40": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본물성_기본물성시험_{sich_code}.xlsx"]["기본물성_기본물성_심도(G L, -m)"]))], "토사_입도분석_체통과백분율#200": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본물성_기본물성시험_{sich_code}.xlsx"]["기본물성_기본물성_심도(G L, -m)"]))], "토사_입도분석_체통과백분율#0.005mm이하": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본물성_기본물성시험_{sich_code}.xlsx"]["기본물성_기본물성_심도(G L, -m)"]))]}
            if sub_test_2:
                for sich_code, info in sub_test_2.copy().items():
                    all_tests = [k.replace(k.split("_")[-1], "")[:-1] for k in info.keys()]

                    for test_name in ['토사_삼축압축_CU', '토사_삼축압축_UU', '토사_압밀시험']:
                        if ('토사_일축압축' in all_tests) and (test_name not in all_tests):
                            new_key = f"{test_name}_{sich_code}.xlsx"

                            if test_name == "토사_삼축압축_CU":
                                sub_test_2[sich_code][new_key] = {"토사_삼축압축CU_심도": [float("nan")], "토사_삼축압축CU_점착력": [float("nan")], "토사_삼축압축CU_내부마찰각": [float("nan")]}
                            elif test_name == "토사_삼축압축_UU":
                                sub_test_2[sich_code][new_key] = {"토사_삼축압축UU_심도": [float("nan")], "토사_삼축압축UU_점착력": [float("nan")]}
                            elif test_name == "토사_압밀시험":
                                sub_test_2[sich_code][new_key] = {"토사_압밀_심도": [float("nan")], "토사_압밀_선행압밀하중": [float("nan")], "토사_압밀_압축지수": [float("nan")]}

        elif main_category == "현장투수 및 수압시험":
            if sub_test_1:
                for sich_code, info in sub_test_1.copy().items():
                    all_tests = [k.replace(k.split("_")[-1], "")[:-1] for k in info.keys()]

                    if ("기본현장_현장투수시험" in all_tests) and ("기본현장_현장수압시험" not in all_tests):
                        new_key = f"기본현장_현장수압시험_{sich_code}.xlsx"
                        sub_test_1[sich_code][new_key] = {"기본현장_현장수압_심도(m)": [depth for depth in range(len(sub_test_1[sich_code][f"기본현장_현장투수시험_{sich_code}.xlsx"]["기본현장_현장투수_심도(m)"]))], "기본현장_현장수압_시간간격": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본현장_현장투수시험_{sich_code}.xlsx"]["기본현장_현장투수_심도(m)"]))], "기본현장_현장수압_수압": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본현장_현장투수시험_{sich_code}.xlsx"]["기본현장_현장투수_심도(m)"]))], "기본현장_현장수압_평균투수계수": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본현장_현장투수시험_{sich_code}.xlsx"]["기본현장_현장투수_심도(m)"]))], "기본현장_현장수압_평균루전값": [float("nan") for _ in range(len(sub_test_1[sich_code][f"기본현장_현장투수시험_{sich_code}.xlsx"]["기본현장_현장투수_심도(m)"]))]}

        elif main_category == "암석시험":
            if sub_test_1:
                for sich_code, info in sub_test_1.copy().items():
                    all_tests = [k.replace(k.split("_")[-1], "")[:-1] for k in info.keys()]

                    for test_name in ['암석_삼축압축', '암석_점하중']:
                        if ('암석_일축압축' in all_tests) and (test_name not in all_tests):
                            new_key = f"{test_name}_{sich_code}.xlsx"

                            if test_name == "암석_삼축압축":
                                sub_test_1[sich_code][new_key] = {"암석_삼축압축_심도": [float("nan")], "암석_삼축압축_점착력": [float("nan")], "암석_삼축압축_내부마찰각": [float("nan")]}
                            elif test_name == "암석_점하중":
                                sub_test_1[sich_code][new_key] = {"암석_점하중_심도": [float("nan")], "암석_점하중_점하중강도": [float("nan")], "암석_점하중_일축압축강도": [float("nan")]}

        # 프롬프트에 생성해야 될 표 개수 입력
        write_test_1 = False
        write_test_2 = False
        write_test_3 = False
        n_tables = 0
        for sub_category_idx, (sub_category, info) in enumerate(sorted_new_all_info.items()):
            if sub_category in ['기본물성_기본물성시험', '토사_입도분석', '기본현장_현장수압시험', '기본현장_현장투수시험', '기본현장_표준관입시험', '암석_삼축압축', '암석_일축압축', '암석_점하중', '물리검층_하향식탄성파']:
                if write_test_1 == False:
                    write_test_1 = True
                    n_tables += 1
            elif sub_category in ['토사_일축압축', '토사_삼축압축_CU', '토사_삼축압축_UU', '토사_압밀시험', '암석_절리면전단']:
                if write_test_2 == False:
                    write_test_2 = True
                    n_tables += 1
            elif sub_category in ['토사_CBR']:
                if write_test_3 == False:
                    write_test_3 = True
                    n_tables += 1
        user_input += f"생성해야 할 표 개수: {n_tables}\n"

        # 입력 프롬프트 작성
        if len(sub_test_1):
            user_input += "\n# 하위시험번호: 1\n"
            for sich_code, sich_code_values in sub_test_1.items():
                user_input += f"'{sich_code}': {sich_code_values}\n"

        if len(sub_test_2):
            user_input += "\n# 하위시험번호: 2\n"
            for sich_code, sich_code_values in sub_test_2.items():
                user_input += f"'{sich_code}': {sich_code_values}\n"

        if len(sub_test_3):
            user_input += "\n# 하위시험번호: 3\n"
            for sich_code, sich_code_values in sub_test_3.items():
                user_input += f"'{sich_code}': {sich_code_values}\n"

        prompt = """### 제공된 예시와 같은 출력을 생성하기 위해 다음 공통 지침과 시험 별 지침을 따르세요.

### 공통 지침
입력으로 시험번호, 주 시험명, 생성해야 할 표 개수, 그리고 지반시험 데이터가 주어집니다. 주 시험명은 '토질시험', '현장투수시험', '표준관입시험', '암석시험', '하향식탄성파' 중 주어집니다.
출력은 시험번호와 주 시험명을 포함하는 헤딩으로 시작되며, '### (시험번호) 주 시험명' 형식을 따릅니다. 그 다음으로 아래 시험 별 지침에 따라 각 시험 유형에 대해 데이터를 요약하는 표를 작성해야 합니다.

### 시험 별 지침
# '토질시험':
하위시험번호 1번에 대한 정보가 주어지면 1번 표를 작성합니다. 1번 표의 열은 ['시추공코드', '심도(m)', 'Wn(%)', 'Gs', 'LL(%)', 'PI', 'No.4(%)', 'No.10(%)', 'No.40(%)', 'No.200(%)', '0.005mm 이하(%)', 'USCS'] 로 구성됩니다. '심도(m)' 열은 주어진 심도 값들을 모두 작성합니다. 'Wn(%)', 'Gs', 'LL(%)', 'PI', 'USCS' 열은 '기본물성_기본물성시험_(시추공코드).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요. 'No.4(%)', 'No.10(%)', 'No.40(%)', 'No.200(%)', '0.005mm 이하(%)' 열은 '토사_입도분석_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요.
하위시험번호 2번에 대한 정보가 주어지면 2번 표를 작성합니다. 2번 표의 열은 ['시추공코드', '심도(m)', '일축압축강도(자연시료)(kgf/cm^2)', '삼축압축(CU) 점착력(kgf/cm^2)', '삼축압축(CU) 내부마찰각(ϕ, °)', '삼축압축(UU) 점착력(C_u, kgf/cm^2)', '압밀 선행압밀하중(PC, kgf/cm^2)', '압밀 압축지수(Cc)'] 로 구성됩니다. '심도(m)' 열은 주어진 심도 값들을 모두 작성합니다. '일축압축강도(자연시료)(kgf/cm^2)' 열은 '토사_일축압축_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요. '삼축압축(CU) 점착력(kgf/cm^2)', '삼축압축(CU) 내부마찰각(ϕ, °)' 열은 '토사_삼축압축_CU_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요. '삼축압축(UU) 점착력(C_u, kgf/cm^2)' 열은 '토사_삼축압축_UU_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요. '압밀 선행압밀하중(PC, kgf/cm^2)', '압밀 압축지수(Cc)' 열은 '토사_압밀시험_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요.
하위시험번호 3번에 대한 정보가 주어지면 3번 표를 작성합니다. 3번 표의 열은 ['시추공코드', '심도(m)', 'A다짐 최대건조밀도(kN/m^3)', 'A다짐 최적함수비(OMC, %)', 'D다짐 최대건조밀도(kN/m^3)', 'D다짐 최적함수비(OMC, %)'] 로 구성됩니다. 모든 열은 '토사_CBR_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성합니다.

# '현장투수 및 수압시험':
1번 표의 열은 ['시추공코드', '심도(m)', '현장투수 평균투수계수(cm/sec)', '시간간격(sec)', '수압(MPa)', '현장수압 평균투수계수(cm/sec)', '평균루전값(l/min/m)'] 로 구성됩니다. '심도(m)' 열은 주어진 심도 값들을 모두 작성합니다. '현장투수 평균투수계수(cm/sec)' 열은 '기본현장_현장투수시험_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요. '시간간격(sec)', '수압(MPa)', '현장수압 평균투수계수(cm/sec)', '평균루전값(l/min/m)' 열은 '기본현장_현장수압시험_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성하세요.

# '표준관입시험':
1번 표의 열은 ['시추공코드', '심도(-m)', 'N값'] 로 구성됩니다. '심도(-m)' 열은 '기본현장_표준관입_표준관입심도(-m)' 의 값을 참고하여 작성합니다. 'N값' 열은 '기본현장_표준관입_표준관입시험' 의 값을 참고하여 작성합니다.

# '암석시험':
하위시험번호 1번에 대한 정보가 주어지면 1번 표를 작성합니다. 1번 표의 열은 ['시추공코드', '심도(m)', '일축압축강도(kgf/cm^2)', '탄성계수(kgf/cm^2)', '삼축압축 점착력(MPa)', '삼축압축 내부마찰각(Φ, °)', '점하중 일축압축강도(MPa)', '점하중강도(MPa)'] 로 구성됩니다. '심도(m)' 열은 주어진 심도 값들을 모두 작성합니다. '일축압축강도(kgf/cm^2)', '탄성계수(kgf/cm^2)' 열은 '암석_일축압축_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성합니다. '삼축압축 점착력(MPa)', '삼축압축 내부마찰각(Φ, °)' 열은 '암석_삼축압축_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성합니다. '점하중 일축압축강도(MPa)', '점하중강도(MPa)' 열은 '암석_점하중_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성합니다.
하위시험번호 2번에 대한 정보가 주어지면 2번 표를 작성합니다. 2번 표의 열은 ['시추공코드', '심도(m)', '절리면 점착력(MPa)', '절리면 내부마찰각(Φ, °)', '절리면압축강도(MPa)', '절리면 수직응력(MPa)', '절리면 전단응력(MPa)'] 로 구성됩니다. 모든 열은 '암석_절리면전단_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성합니다.

# '하향식탄성파':
1번 표의 열은 ['시추공코드', '심도(m)', '전단파속도 P파(m/s)', '전단파속도 S파(m/s)', '포아송비(υ)', '전단탄성계수(MPa)', '영률(MPa)', '밀도(g/m^3)'] 로 구성됩니다. 모든 열은 '물리검층_하향식탄성파_(시추공번호).xlsx' 의 정보에서 매칭되는 키의 값을 참고하여 작성합니다.
"""

        base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        lora_adapter_path = os.path.join(os.path.dirname(__file__), "..", "report_llm", "llm", "report_3", "qwen_merged_part3")
        merged_model_path = os.path.join(os.path.dirname(__file__), "..", "report_llm", "llm", "report_3", "qwen_part3_merged_final")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='auto',
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            cache_dir=merged_model_path
        )
        # model = AutoModelForCausalLM.from_pretrained(
        #     lora_adapter_path,
        #     device_map='auto',
        #     torch_dtype=torch.float16,
        #     quantization_config=quantization_config
        # )
        model.eval()

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        
        user_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([user_input], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response_3 += f"{response}\n"

    return response_3.strip()
           
def create_report(retrieval_distance, max_retrieval, retrieval_option):
    '''사용자 입력 좌표로부터 {retrieval_distance}(m) 내의 시추공 데이터 중에서 0 ~ {max_retrieval}개를 참고하여 보고서 생성'''
    
    with st.spinner("Finding sichs..."):
        retrieved_sichs = find_sichs_within_radius(retrieval_distance, max_retrieval, retrieval_option)

        all_file_paths = {}
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset")

        for top_folder in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, top_folder)):
                for pj in os.listdir(os.path.join(dataset_path, top_folder)):
                    if os.path.isdir(os.path.join(dataset_path, top_folder, pj)):
                        for sich in os.listdir(os.path.join(dataset_path, top_folder, pj)):
                            all_file_paths[sich] = os.path.abspath(os.path.join(dataset_path, top_folder, pj, sich))

        retrieved_sich_paths = {key: all_file_paths[key] for key, value in retrieved_sichs.items()}

    with st.spinner("Preprocessing data..."):
        text_1_2 = preprocess_1_2(retrieved_sich_paths)
        all_test = preprocess_3(retrieved_sich_paths)

    with st.spinner("Generating report..."):
        response_1_2 = get_response_1_2(text_1_2)
        response_3 = get_response_3(all_test)

    report = response_1_2
    if response_3:
        report += "\n\n## [부록] 지반조사 결과\n\n" + response_3
    return report


class RAGChatbot:
    def __init__(self, chunks, retriever_model_name, generator_model_name):
        self.chunks = chunks
        self.retriever_model_name = retriever_model_name
        self.generator_model_name = generator_model_name

        self.vector_store = None
        self.embedding_model = None

        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_devices is not None:
            self.device_ids = list(map(int, cuda_devices.split(',')))  # ex: '0,1' -> [0, 1]
            self.device_map = {f'cuda:{i}': f'cuda:{self.device_ids[i]}' for i in range(len(self.device_ids))}
        else:
            self.device_ids = None
            self.device_map = {'cpu': 'cpu'}


    def build_index(self, store_name=os.path.join(os.path.dirname(__file__), "..", "qa_llm", "retriever", "faiss_index")):
        """청크 임베딩 저장"""

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "qa_llm", "retriever")):
            os.makedirs(os.path.join(os.path.dirname(__file__), "..", "qa_llm", "retriever"))

        if self.retriever_model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
            from langchain_openai import OpenAIEmbeddings
            from langchain.storage import LocalFileStore
            from langchain_community.vectorstores import FAISS
            from langchain.embeddings import CacheBackedEmbeddings

            load_dotenv()
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

            # 임베딩 생성 (캐시 적용)
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            store = LocalFileStore(os.path.join(os.path.dirname(__file__), "..", "qa_llm", "retriever", "cache"))
            cached_embedder = CacheBackedEmbeddings.from_bytes_store(self.embedding_model, store, namespace=self.embedding_model.model)

            # FAISS 인덱스 생성
            vector_store = FAISS.from_texts(self.chunks, cached_embedder)
            vector_store.save_local(store_name)
            self.vector_store = FAISS.load_local(store_name, self.embedding_model, allow_dangerous_deserialization=True)
            print(f"FAISS index built with {self.vector_store.index.ntotal} chunks. => 'qa_llm/{store_name}'")

        elif self.retriever_model_name == "BAAI/bge-m3":
            from FlagEmbedding import BGEM3FlagModel, FlagReranker

            # 임베딩 생성
            embedding_model_path = os.path.join(os.path.dirname(__file__), "..", "qa_llm", "retriever", "bge-m3")
            self.embedding_model = BGEM3FlagModel(
                'BAAI/bge-m3',
                device_map=self.device_map,
                use_fp16=True,
                cache_dir=embedding_model_path,
            )

            # 로컬 저장 (FAISS 미지원 모델)
            embedding_save_path = os.path.join(os.path.dirname(__file__), "..", "qa_llm", "retriever", "bge-m3_embeddings.npz")
            if not os.path.exists(embedding_save_path):
                chunks_embedding = self.embedding_model.encode(self.chunks)["dense_vecs"]
                np.savez(embedding_save_path, embeddings=chunks_embedding)
                print("Embeddings is saved to 'qa_llm/retriever/bge-m3_embeddings.npz'")

        else:
            raise Exception("Invalid retriever model name")


    def search(self, query: str, top_k: int = 5):
        """쿼리 텍스트와 가장 유사한 청크 검색"""

        if self.retriever_model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            contexts = retriever.invoke(query)
            contexts = "\n".join([doc.page_content for doc in contexts])

        elif self.retriever_model_name == "BAAI/bge-m3":
            query_embedding = self.embedding_model.encode(query, batch_size=top_k, max_length=500)['dense_vecs']
            chunks_embedding = np.load(os.path.join(os.path.dirname(__file__), "..", "qa_llm", "retriever", "bge-m3_embeddings.npz"))["embeddings"]

            similarity = query_embedding @ chunks_embedding.T
            top_k_indices = np.argsort(similarity)[-top_k:].tolist()
            contexts = "\n".join([self.chunks[i] for i in top_k_indices])
            
        return contexts


    def generate_response(self, query: str, report: str, top_k: int = 5):
        """쿼리에 대한 응답 생성"""

        # 1. 관련 문서 검색
        contexts = self.search(query, top_k=top_k)

        # 2. contexts를 기반으로 모델 입력 구성
        prompt = [
            {"role": "system", "content": f"### 지침\n아래 주어진 지반조사보고서, 사용자 질문과 관련된 컨텍스트를 참고하여 사용자 질문에 대해 답변해주세요.\n\n### 지반조사보고서\n{report}\n\n### 컨텍스트\n{contexts}\n\n"},
            {"role": "user", "content": query}
        ]

        # 3. 모델 응답 생성
        generator_model_path = os.path.join(os.path.dirname(__file__), "..", "qa_llm", "generator", "MLP-KTLim", "llama-3-Korean-Bllossom-8B")
        tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name, padding=True, truncation=True, trust_remote_code=True, cache_dir=generator_model_path)

        model = AutoModelForCausalLM.from_pretrained(
            self.generator_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            resume_download=True,
            trust_remote_code=True,
            cache_dir=generator_model_path
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        
        input_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)