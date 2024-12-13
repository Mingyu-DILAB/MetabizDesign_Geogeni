import os
import sys
import json
import glob
import openai
import argparse
import camelot
import numpy as np
import pandas as pd
import markdown2
import pdfkit
import random
import itertools
import copy
import tiktoken
import pyproj
import math
import requests
from geopy.geocoders import Nominatim

from typing import List
from pprint import pprint
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv
from tqdm import tqdm
from utils import CustomPrompts

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
random.seed(24)

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--input_path', 
        required=True,
        nargs='+',
        help='Multiple input -> List'
    )
    p.add_argument(
        '--output_path', 
        required=True,
        help=''
    )
    args = p.parse_args()
    return args

def num_tokens_from_string(text):
    """Returns the number of tokens in a text."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def partition_list(lst):
    lst_ = copy.deepcopy(lst)
    sub_lst = []
    
    while True:
        if len(lst_) == 0:
            break
        
        numbers = list(range(1, min(len(lst_)+1, 11)))
        weights = [2**i if i<=5 else 32//(2**(i-6)) for i in range(1, min(len(lst_)+1, 11))]
        # num_to_select = random.choices(numbers, k=1)[0]
        num_to_select = random.choices(numbers, weights=weights, k=1)[0]
        
        random_lst = random.sample(lst_, k=num_to_select)
        sub_lst.append(random_lst)
        for x in random_lst:
            lst_.remove(x)
        
    return sub_lst

def generate_report(n_sichs, text):
    try:
        print("Generating report...", end=" ")
        responses = {"body": {}, "appendix": {}}
        
        # 1번, 2번(body)에 대한 보고서 생성
        prompt_1_2 = CustomPrompts().prompt_1_2()
        
        llm_1_2 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, seed=24, max_tokens=None)
        chain_1_2 = prompt_1_2 | llm_1_2 | StrOutputParser()
        response_1_2 = chain_1_2.invoke({"text": text})
        
        responses["body"] = {"text": text, "response": response_1_2}
        print("Done", end="\n\n")
        
        return responses
    
    except Exception as ex:
        print(f"OpenAI API call error: {ex}")
        return None

# def save_markdown_to_pdf(args, markdown_text, output_file_name):
#     from markdown_pdf import MarkdownPdf, Section
#     pdf = MarkdownPdf(toc_level=2)
    
#     n_sichs = str(output_file_name.count("_") + 1)
#     if not os.path.exists(os.path.join(args.output_path, n_sichs)):
#         os.makedirs(os.path.join(args.output_path, n_sichs))
#     output_pdf_path = os.path.join(args.output_path, n_sichs, output_file_name)
    
#     pdf.add_section(Section(markdown_text))
#     pdf.save(output_pdf_path)
#     print(f"Markdown has been converted to PDF and saved as {output_pdf_path}")
    
def save_markdown_to_json(args, report, output_file_name):
    n_sichs = str(output_file_name.count("_") + 1)
    if not os.path.exists(os.path.join(args.output_path, n_sichs)):
        os.makedirs(os.path.join(args.output_path, n_sichs))

    # body
    output_body_path = os.path.join(args.output_path, n_sichs, "body.json")
    body = pd.DataFrame.from_dict([report["body"]])

    if os.path.exists(output_body_path):
        with open(output_body_path, 'r', encoding='utf-8') as f:
            existing_json = pd.read_json(f, orient='records', lines=True)
        combined_json = pd.concat([existing_json, body], ignore_index=True)
    else:
        combined_json = body
    combined_json.to_json(output_body_path, orient="records", lines=True, force_ascii=False)
    
def save_markdown_to_md(args, report, output_file_name):
    n_sichs = str(output_file_name.count("_") + 1)
    if not os.path.exists(os.path.join(args.output_path, n_sichs, "md")):
        os.makedirs(os.path.join(args.output_path, n_sichs, "md"))
    
    # body
    body = report["body"]["response"]
    
    output_body_name = output_file_name.split(".")[0] + "_body.md"
    output_path = os.path.join(args.output_path, n_sichs, "md", output_body_name)
    with open(output_path, "w") as f:
        f.write(body)

class Chunking():
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
        with open(os.path.join(os.path.dirname(__file__), "..", "dataset_v2", "pjCoord2.json"), "r", encoding="utf-8") as pjCoord_file:
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
    
    def get_docs(self, sich_path=None) -> List[Document]:
        if self.extension == "pdf":
            pdf_docs = self._pdf_load()
            return pdf_docs
        
        elif self.extension == "xlsx":
            xlsx_docs = self._xlsx_load()
            return xlsx_docs
        
        elif self.extension == "json":
            json_docs = self._json_load(sich_path)
            return json_docs


def main(args):
    error_files = []

    for directory in args.input_path:
        for root, dirs, files in os.walk(directory):
            dirs = sorted(dirs)
            relative_path = os.path.relpath(root, directory)
            path_depth = len(relative_path.split(os.sep))
            if path_depth == 1:
                for subdir in tqdm(dirs):
                    # 삭제
                    subdir_path = os.path.join(root, "F0545")
                    # subdir_path = os.path.join(root, subdir)
                    project_meta_json = subdir + "_project_meta.json"

                    if project_meta_json in os.listdir(subdir_path):
                        sichs_list = [x for x in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, x))]
                        sichs_list = partition_list(sichs_list)
                        
                        for sichs in sichs_list:
                            # 삭제
                            sichs = ["F0545BH002"]

                            sichs = sorted(sichs)

                            print(sichs)
                            print("Generating context...", end=" ")
                            text = """"""
                            error = False

                            # 시험 개수 직접 카운트(2-2 시험 DB 내역에 활용)                        
                            n_test = {"기본물성_기본물성시험": 0, "기본현장_표준관입시험": 0, "기본현장_현장수압시험": 0, "기본현장_현장투수시험": 0, "물리검층_하향식탄성파": 0, "암석_일축압축": 0, "토사_입도분석": 0}
                            for i, sich in enumerate(sichs):
                                sich_path = os.path.join(subdir_path, sich)

                                for test in os.listdir(sich_path):
                                    test_path = os.path.join(sich_path, test)
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
                            for i, sich in enumerate(sichs):
                                try:
                                    with open(os.path.join(subdir_path, sich, "기본현장_표준관입시험", "기본현장_표준관입시험" + f"_{sich}.json"), "r") as f:
                                        layers = json.load(f)["LAYER"]

                                        for layer in layers:
                                            if layer["name"] not in stratum:
                                                stratum[layer["name"]] = {
                                                    "층후": {"from": layer["from"], "to": layer["to"]},
                                                    "desc": [f"시추공 {sich}: {layer['desc']}"]
                                                }
                                            else:
                                                if layer["from"] < stratum[layer["name"]]["층후"]["from"]:
                                                    stratum[layer["name"]]["층후"]["from"] = layer["from"]
                                                
                                                if layer["to"] > stratum[layer["name"]]["층후"]["to"]:
                                                    stratum[layer["name"]]["층후"]["to"] = layer["to"]

                                                stratum[layer["name"]]["desc"].append(f"시추공 {sich}: {layer['desc']}")

                                # 기본현장_표준관입시험 폴더가 없는 경우
                                except:
                                    pass
                            
                            def get_layer_type_priority(key):
                                return 0 if "토층" in key else 1

                            try:
                                stratum = dict(sorted(stratum.items(), key=lambda x: (get_layer_type_priority(x[0]), x[1]['층후']['from'], x[1]['층후']['to'])))
                            except:
                                continue

                            # Context
                            context = {"시험 DB 내역": n_test_category, "지층 개요": stratum}
                            context = json.dumps(context, ensure_ascii=False)
                            context = context.replace("{", "[").replace("}", "]")
                            text += f"""# Context\n{context}\n\n"""

                            # Each sich context
                            each_sich_context = {}
                            for i, sich in enumerate(sichs):
                                # sich_meta
                                sich_meta_path = os.path.join(subdir_path, sich, sich + "_sich_meta.json")
                                if os.path.exists(sich_meta_path):
                                    with open(sich_meta_path, "r", encoding="utf-8") as f:
                                        sich_meta = json.load(f)

                                    each_sich_context = {k: v for k, v in sich_meta.items() if k in ["조사명", "구분", "기간", "시추공명", "시추공코드", "심도", "표고", "지하수위"]}

                                # coordinates, address
                                sich_path = os.path.join(subdir_path, sich)
                                for test in os.listdir(sich_path):
                                    test_path = os.path.join(sich_path, test)
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
                                                            # chunk = Chunking(file_path=file_path, extension=extension).get_docs()
                                                        elif extension == "json":
                                                            if test == "기본현장_표준관입시험":
                                                                chunk = Chunking(file_path=file_path, extension=extension).get_docs(sich_path)
                                                                for k, v in chunk.items():
                                                                    each_sich_context[k] = chunk[k]
                                                            
                                                    except Exception as ex:
                                                        print(f"Error processing file {file_path}: {ex}")
                                                        error_files.append(file_path)
                                                        error = True
                                                else:
                                                    print(f"Skipping directory (not a file): {file_path}")
                                
                                each_sich_context = json.dumps(each_sich_context, ensure_ascii=False)
                                each_sich_context = each_sich_context.replace("{", "[").replace("}", "]")

                                text += f"""## 시추공 {i+1}: {sich}\n{each_sich_context}\n\n"""
                            
                            if error: continue
                            print("Done")

                            # generation
                            if text:
                                try:
                                    report = generate_report(n_sichs=len(sichs), text=text.strip())
                                    sichs = str(sichs).replace("[","").replace("]","").replace("'","").replace(", ", "_")
                            
                                    if report:
                                        output_file_name = f"{os.path.basename(str(sichs))}.pdf"
                                        save_markdown_to_md(args, report, output_file_name)
                                        save_markdown_to_json(args, report, output_file_name)
                                    else:
                                        print("Report generation failed.")
                                        error_files.append(str(sichs))
                                except Exception as ex:
                                    print(f"Error generating report for directory {str(sichs)}: {ex}")
                                    error_files.append(str(sichs))

                            sys.exit(0)
                        
    if error_files:
        print("These files or directories had errors:")
        for error_file in error_files:
            print(error_file)
    

if __name__ == "__main__":
    args = define_argparser()
    main(args)