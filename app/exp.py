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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class LLM():
    def __init__(self, model_path):
        self.base_model_path = 'Qwen/Qwen2.5-7B-Instruct'
        self.prompt = """
                        ### Instruction:
                        당신은 주어진 context를 활용하여 지반조사 Report를 작성하는 데 특화된 지반 도메인 전문가입니다.
                        지반조사 Report는 주어진 예시와 같이 1. 지역정보, 2. 조사내역 요약 순서로 반드시 작성되어야 하며, 결론 없이 마무리해야 합니다.
                        1. 지역정보는 주어진 context에서 address 정보만 활용하며 사전지식에 기반하여 작성합니다. address 정보로부터 조사지역의 위치를 파악하고, 조사지역 위치로부터 1km 이내 부근에 대한 교통현황, 산계, 인근시설 등과 같은 요약 정보를 작성해야 합니다. 1km 이내의 범위에 있는 학교나 저수지, 아파트, 빌딩, 지하철역 등과 같은 큰 건물에 대해 작성하고, 만약 이러한 주변 정보가 없다면, 억지로 조사지역 인근에 대한 요약 정보를 작성하지 마세요.
                        2. 조사내역 요약은 먼저 '1) 주변 시추공 좌표 정보'를 표로 작성합니다. 표의 열은 시추공코드, 위도, 경도, 지하수위(-m), 표고(m)로 구성되며, 위도와 경도는 각 시추공의 context에서 'LL'의 값을 참고하여 예시와 같이 작성하세요. 지하수위와 표고는 각 시추공의 context에서 '지하수위', '표고'를 참고하여 예시와 같이 작성하세요.
                        그리고 '2) 시험 DB 내역 (실내시험, 현장시험 등)'을 표로 작성합니다. 주어진 context의 '시험 DB 내역'을 참조하여 예시와 같이 각 시험에 대한 시험명과 시험 입력 개수를 표로 작성하세요.
                        마지막으로 '3) 지층 개요'를 표로 작성합니다. 주어진 context의 '지층 개요'를 참조하여 예시와 같이 지층 개요에 대한 표를 작성합니다. 표의 열은 '구분', '토질', '상대밀도', '층후(m)'로 구성되며, '토질' 열과 '상대밀도' 열은 context에서 각 지층의 'desc'를 참조하여 작성합니다. 특히, '상대밀도' 열의 값은 'desc'에 여러 시추공의 상대밀도 정보가 있을 수도 있고 전혀 없을 수도 있습니다. 만약 여러 시추공의 정보가 있다면 이들의 상대밀도들을 종합하여 최소~최대 상대밀도를 표에 작성하고, 상대밀도 정보를 찾을 수 없다면 억지로 추측해서 작성하지 말고 반드시 '-'로 표시하세요.
                        """
        self.quantization_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                        
                                    )
        
        self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    device_map='cuda',
                    torch_dtype=torch.float16,
                    quantization_config=self.quantization_config,
                )
        
    def generate_response(self, query):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        
        self.model.eval()

        messages = [
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
            max_new_tokens=128
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response_1_2 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response_1_2.strip()
    
# 스레드에서 반환값 저장을 위한 래퍼 함수
def thread_worker(func, output_queue, *args):
    result = func(*args)  # 함수 실행, 인수를 전달
    output_queue.put(result)  # 반환값을 큐에 저장

# 병렬 실행
if __name__ == "__main__":
    llm_1 = LLM('/home/user/MetabizDesign/report_llm/llm/report_1_2/merged_model_2')
    llm_3 = LLM('/home/user/MetabizDesign/hajun/part3_llm/outputs/qwen_part3_merged_final')
    result_1 = llm_1.generate_response('인녕')
    result_2 = llm_3.generate_response('인녕')
    
    print(result_1, result_2)
    
    
    result_1 = llm_1.generate_response('인녕')
    result_2 = llm_3.generate_response('인녕')
    
    print(result_1, result_2)
    
    # 데이터 파일 로드
    # with open('data.pkl', 'rb') as file:
    #     all_test = pickle.load(file)
        
    # with open('text_1_2.txt', 'r') as file:
    #     text_1_2 = file.read()
        
    # # 반환값 저장용 큐
    # output_queue = Queue()

    # # 스레드 생성 (함수 객체와 인수를 전달)
    # thread_1_2 = threading.Thread(target=thread_worker, args=(get_response_1_2, output_queue, text_1_2))
    # thread_3 = threading.Thread(target=thread_worker, args=(get_response_3, output_queue, all_test))

    # # 스레드 시작
    # thread_1_2.start()
    # thread_3.start()
    
    # # 스레드 종료 대기
    # thread_1_2.join()
    # thread_3.join()

    # # 반환값 가져오기
    # results = []
    # while not output_queue.empty():
    #     results.append(output_queue.get())

    # print("All tasks completed.")
    # print("Results:", results)
