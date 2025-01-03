import os
import sys
import cv2
import ast
import json
import random
import base64
import argparse
import numpy as np
import logging

from openai import OpenAI
from pdf2image import convert_from_path
from dotenv import load_dotenv
from paddleocr import PaddleOCR

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
logging.getLogger('ppocr').setLevel(logging.ERROR)
client = OpenAI()

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--json_file_path', 
        required=True,
    )
    p.add_argument(
        '--pdf_file_path', 
        required=True,
    )
    p.add_argument(
        '--output_path', 
        required=True,
    )
    p.add_argument(
        '--seed', 
        default=42,
    )

    args = p.parse_args()
    return args

def seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def load_pdf_page_as_image(page_index, pdf_path):
    pages = convert_from_path(pdf_path)
    return np.array(pages[page_index])

def scale_bbox(bbox, pdf_size, image_size):
    '''PDF 좌표를 이미지 좌표로 변환'''
    pdf_width, pdf_height = pdf_size
    img_width, img_height = image_size

    x_scale = img_width / pdf_width
    y_scale = img_height / pdf_height

    x_min = int(bbox[0] * x_scale)
    y_min = int(bbox[1] * y_scale)
    x_max = int(bbox[2] * x_scale)
    y_max = int(bbox[3] * y_scale)

    return x_min, y_min, x_max, y_max

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_description(image_path, data_type):
    while True:
        if os.path.exists(image_path):
            break

    base64_image = encode_image(image_path)

    if data_type == "table":
        instruction = f"당신은 PDF 문서에서 추출된 테이블을 입력으로 받아, 테이블의 내용이 담긴 청크 텍스트를 출력하는 테이블 분석 전문가입니다. 청크 텍스트는 줄글 형태로 작성해야 하며, 테이블 내용 이외의 부연설명은 작성하지 마세요."

    elif data_type == "image":
        instruction = f"당신은 PDF 문서에서 추출된 이미지를 입력으로 받아, 이미지의 내용이 담긴 청크 텍스트를 출력하는 이미지 분석 전문가입니다. 청크 텍스트는 줄글 형태로 작성해야 하며, 이미지 내용 이외의 부연설명은 작성하지 마세요."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=None,
    )

    return response.choices[0].message.content

def main(args):
    with open(args.json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    chunks = ""
    for page in json_data['pdf_info']:
        page_index = page['page_idx']
        title = ""
        n_titles = 0

        print(f"Processing page {page_index+1}...")

        page_image = load_pdf_page_as_image(page_index, args.pdf_file_path)

        pdf_width, pdf_height = page['page_size']
        image_height, image_width, _ = page_image.shape

        for i, block in enumerate(page.get('preproc_blocks', [])):
            data_type = block["type"]

            if len(block["bbox"]) == 4:
                x_min, y_min, x_max, y_max = scale_bbox(block['bbox'], (pdf_width, pdf_height), (image_width, image_height))

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image_width, x_max)
                y_max = min(image_height, y_max)

                cropped_image = page_image[y_min:y_max, x_min:x_max]

                save_dir = os.path.join(args.output_path, args.json_file_path.split(os.sep)[-4], args.json_file_path.split(os.sep)[-3], data_type)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                image_path = os.path.join(save_dir, f"{page_index+1}p_{i+1-n_titles}.png")
                cv2.imwrite(image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                try:
                    if data_type in ["title"]:
                        # PaddleOCR
                        ocr = PaddleOCR(lang="korean")
                        result = ocr.ocr(image_path, cls=True)
                        text = " ".join([line[-1][0] for line in result[0]])

                        if not title:
                            title = text
                        else:
                            title = f"{title} -> {text}"

                        n_titles += 1
                    
                    elif data_type in ["text"]:
                        # PaddleOCR
                        ocr = PaddleOCR(lang="korean")
                        result = ocr.ocr(image_path, cls=True)
                        text = " ".join([line[-1][0] for line in result[0]])
                        
                        chunk = {
                            "text": text,
                            "filename": args.json_file_path.split(os.sep)[-1].split("_middle.json")[0] + ".pdf",
                            "page": page_index+1,
                            "coordinates": (x_min, y_min, x_max, y_max),
                            "order": i+1-n_titles,
                            "data_type": data_type
                        }

                        # 이전 청크가 title이라면, 텍스트 앞에 title 추가
                        if title: chunk['text'] = f"제목: {title}, 내용: {chunk['text']}"
                        title = ""

                        chunks += f"{chunk}\n"

                        print("="*100)
                        print(chunk)
                        print("="*100)
                        print()

                    elif data_type in ["table", "image"]:
                        description = generate_image_description(image_path, data_type)

                        chunk = {
                            "text": description,
                            "filename": args.json_file_path.split(os.sep)[-1].split("_middle.json")[0] + ".pdf",
                            "page": page_index+1,
                            "coordinates": {"points": (x_min, y_min, x_max, y_max)},
                            "order": i+1-n_titles,
                            "data_type": data_type
                        }
                        chunks += f"{chunk}\n"

                        print("="*100)
                        print(chunk)
                        print("="*100)
                        print()
                
                except:
                    print("="*100)
                    print(image_path)
                    print("None")
                    print("="*100)
                    print()
                    pass

    chunks_save_path = os.path.join(args.output_path, args.json_file_path.split(os.sep)[-4], args.json_file_path.split(os.sep)[-3], "chunks.txt")
    with open(chunks_save_path, "w", encoding="utf-8") as file:
        file.write(chunks)
        print(f"청크 파일 저장완료 => '{chunks_save_path}'")
    print()
    

if __name__ == "__main__":
    args = define_argparser()
    seed(args.seed)

    main(args)