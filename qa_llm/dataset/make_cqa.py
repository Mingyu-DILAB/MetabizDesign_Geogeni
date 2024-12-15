import os, sys, json, ast
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def load_chunks():
    documents_path = os.path.join(os.path.dirname(__file__), "MinerU", "MyOCR", "results", "mineru_output")

    chunks = []
    for root, dirs, files in os.walk(documents_path):
        if "chunks.txt" in files:
            with open(os.path.join(root, "chunks.txt"), "r", encoding="utf-8") as chunk_file:
                lines = chunk_file.readlines()
                for line in lines:
                    chunks.append(line.strip())
            continue

    return chunks

def generate_cqa(chunk: str):
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"아래 주어진 Chunk를 참고하여 Question과 Answer를 1개만 생성하세요. 유용한 Question과 Answer를 만들어주세요."},
            {"role": "user", "content": f"Chunk:\n{chunk}\n\nQuestion:\n\nAnswer:"}
        ],
        temperature=0.7,
        max_tokens=500
    )

    content = completion.choices[0].message.content
    return parse_cqa(chunk, content)

def parse_cqa(chunk: str, content: str):
    lines = content.split("\n")
    question = None
    answer = None

    try:
        for line in lines:
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
        
        if question and answer:
            return {"Chunk": chunk, "Question": question, "Answer": answer}
        else:
            return None
    except:
        return None

def save_cqa_to_json(cqa, json_file_path):
    try:
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []

        existing_data.append(cqa)

        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
        print(f"CQA data successfully added to {json_file_path}")

    except Exception as e:
        print(e)

def main():
    chunks = load_chunks()

    # 청크의 개수만큼 cqa 생성
    for chunk in tqdm(chunks):
        if len(ast.literal_eval(chunk)["text"]) < 20:
            continue

        cqa = generate_cqa(chunk)

        if cqa:
            save_cqa_to_json(cqa, json_file_path=os.path.join(os.path.dirname(__file__), "cqa.json"))

if __name__ == "__main__":
    main()