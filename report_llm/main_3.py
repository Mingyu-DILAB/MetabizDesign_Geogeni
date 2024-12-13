import os, sys, json, copy, random
import math
import glob
import argparse
# import openai
# import camelot
import pyproj
import tiktoken
import itertools
import numpy as np
import pandas as pd

from collections import OrderedDict
from openai import OpenAI
from typing import List
from langchain.schema import Document
from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv
from tqdm import tqdm
from utils import CustomPrompts
from pprint import pprint

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

def num_tokens_from_string(text: str) -> int:
    """Returns the number of tokens in a text."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def partition_list(lst) -> List:
    '''리스트를 하위 리스트들로 분할'''
    
    lst_ = copy.deepcopy(lst)
    sub_lst = []
    
    while True:
        if len(lst_) == 0:
            break
        
        numbers = list(range(1, min(len(lst_)+1, 11)))
        weights = [2**i if i<=5 else 32//(2**(i-6)) for i in range(1, min(len(lst_)+1, 11))] # 1~10개 중 각각 뽑힐 확률 차등 부여
        # num_to_select = random.choices(numbers, k=1)[0]
        num_to_select = random.choices(numbers, weights=weights, k=1)[0]
        
        random_lst = random.sample(lst_, k=num_to_select)
        sub_lst.append(random_lst)
        for x in random_lst:
            lst_.remove(x)
        
    return sub_lst

def generate_report(text: str):
    messages = CustomPrompts().prompt_3(text)

    try:
        # 3번(appendix)에 대한 보고서 생성
        messages = CustomPrompts().prompt_3(text)

        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=None,
            seed=42,
            temperature=0.1,
        )

        assistant_reply = completion.choices[0].message.content

        responses = {"text": text, "response": assistant_reply.strip()}

        return responses
    
    except Exception as ex:
        print(f"OpenAI API call error: {ex}")
        return None
    
def save_markdown_to_json(args, report, save_name):
    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path))

    save_path = os.path.join(args.output_path, "appendix.json")
    appendix = pd.DataFrame.from_dict([report])

    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            existing_json = pd.read_json(f, orient='records', lines=True)
        combined_json = pd.concat([existing_json, appendix], ignore_index=True)
    else:
        combined_json = appendix
    combined_json.to_json(save_path, orient="records", lines=True, force_ascii=False)
    
def save_markdown_to_md(args, report, save_name):
    if not os.path.exists(os.path.join(args.output_path, "md")):
        os.makedirs(os.path.join(args.output_path, "md"))

    save_path = os.path.join(args.output_path, "md", save_name + ".md")
    with open(save_path, "w") as f:
        f.write(report["response"])

class Chunking():
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


def main(args):
    for directory in args.input_path:
        # 프로젝트 레벨
        for project_code in sorted(os.listdir(directory)):
            project_path = os.path.join(directory, project_code)

            sichs_list = [sich_code for sich_code in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, sich_code))]
            sichs_list = partition_list(sichs_list)

            # 시추공 레벨
            for sichs in sichs_list:
                sichs = sorted(sichs)
                print(sichs)
                print("Getting test info...", end=" ")

                # all_test 변수에 모든 시험 정보들 저장
                all_test = {"토질시험": {sich: {} for sich in sichs}, "현장투수 및 수압시험": {sich: {} for sich in sichs}, "표준관입시험": {sich: {} for sich in sichs}, "암석시험": {sich: {} for sich in sichs}, "하향식탄성파": {sich: {} for sich in sichs}}
                for i, sich in enumerate(sichs):
                    sich_path = os.path.join(project_path, sich)
                    
                    # 시험 레벨
                    for test_name in os.listdir(sich_path):
                        test_path = os.path.join(sich_path, test_name)

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
                                        all_test[main_category][sich][file] = Chunking(file_path=file_path, extension=extension).get_docs()
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
                    print("Error in adding nan values.")
                    continue
                print("Done")

                # 메인 카테고리 별로 보고서 생성
                for category_idx, (main_category, all_info) in enumerate(all_test.items()):
                    print(f"Generating report ({category_idx+1}) {main_category}...", end=" ")

                    text = """"""
                    text += f"### 시험번호: {category_idx+1}\n"
                    text += f"주 시험명: {main_category}\n"

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
                    text += f"생성해야 할 표 개수: {n_tables}\n"

                    # 입력 프롬프트 작성
                    if len(sub_test_1):
                        text += "\n# 하위시험번호: 1\n"
                        for sich_code, sich_code_values in sub_test_1.items():
                            text += f"'{sich_code}': {sich_code_values}\n"

                    if len(sub_test_2):
                        text += "\n# 하위시험번호: 2\n"
                        for sich_code, sich_code_values in sub_test_2.items():
                            text += f"'{sich_code}': {sich_code_values}\n"

                    if len(sub_test_3):
                        text += "\n# 하위시험번호: 3\n"
                        for sich_code, sich_code_values in sub_test_3.items():
                            text += f"'{sich_code}': {sich_code_values}\n"

                    # 이미 저장된 데이터가 있다면 skip
                    sichs = str(sichs).replace("[","").replace("]","").replace("'","").replace(", ", "_")
                    save_name = sichs + "_" + str(category_idx+1) + "_" + main_category
                    if os.path.exists(os.path.join(args.output_path, "md", f"{save_name}.md")):
                        continue
                    
                    # generation
                    text = str(text).replace("{", "[").replace("}", "]")
                    text = text.strip()
                    if text:
                        report = generate_report(text)
                        
                        if report:
                            save_markdown_to_md(args, report, save_name)
                            save_markdown_to_json(args, report, save_name)
                            
                    print("Done")
                print()

if __name__ == "__main__":
    args = define_argparser()
    main(args)