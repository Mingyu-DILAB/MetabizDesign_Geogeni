from collections import OrderedDict
import os
import pickle
from queue import Queue
import threading
import torch
import tiktoken
import logging
import pyproj
import requests
import numpy as np
import pandas as pd
import streamlit as st
import time
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel



os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLM():
    def __init__(self, model_path, prompt):
        self.base_model_path = 'Qwen/Qwen2.5-7B-Instruct'
        self.prompt = prompt
        self.quantization_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                        
                                    )
        
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map='cuda',
                    torch_dtype=torch.float16
                    # quantization_config=self.quantization_config,
                )
        
    def get_response_1_2(self, query):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        
        self.model.eval()

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": query}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response_1_2 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response_1_2.strip()
    
    
    def get_response_3(self, all_test):
        ALL_TEST = {
            "토질시험": ["기본물성_기본물성시험", "토사_입도분석", "토사_일축압축", "토사_삼축압축_CU", "토사_삼축압축_UU", "토사_압밀시험", "토사_CBR"],
            "현장투수 및 수압시험": ["기본현장_현장수압시험", "기본현장_현장투수시험"],
            "표준관입시험": ["기본현장_표준관입시험"],
            "암석시험": ["암석_삼축압축", "암석_일축압축", "암석_절리면전단", "암석_점하중"],
            "하향식탄성파": ["물리검층_하향식탄성파"]
        }
        
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
    


            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self.model.eval()

            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_input}
            ]
            
            user_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([user_input], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response_3 += f"{response}\n"

        return response_3.strip()
    

