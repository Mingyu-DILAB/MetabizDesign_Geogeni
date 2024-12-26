from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import time

# 두 개의 모델 로드
model_name1 = "Qwen/Qwen2.5-7B-Instruct"
model_name2 = "Qwen/Qwen2.5-7B-Instruct"  # 예: GPT-2 모델로 대체 가능

tokenizer1 = AutoTokenizer.from_pretrained(model_name1, trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained(model_name1, device_map="auto", trust_remote_code=True)

tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForCausalLM.from_pretrained(model_name2, device_map="auto")

# 모델 1의 쿼리 처리 함수
def process_query_with_model1(query):
    inputs = tokenizer1(query, return_tensors="pt").to(model1.device)
    outputs = model1.generate(**inputs, max_new_tokens=100)
    result = tokenizer1.decode(outputs[0], skip_special_tokens=True)
    return f"Model 1: {result}"

# 모델 2의 쿼리 처리 함수
def process_query_with_model2(query):
    inputs = tokenizer2(query, return_tensors="pt").to(model2.device)
    outputs = model2.generate(**inputs, max_new_tokens=100)
    result = tokenizer2.decode(outputs[0], skip_special_tokens=True)
    return f"Model 2: {result}"

# 두 모델로 동시에 쿼리를 처리
def process_queries_with_two_models(queries1, queries2):
    with ThreadPoolExecutor() as executor:
        results1 = executor.map(process_query_with_model1, queries1)
        results2 = executor.map(process_query_with_model2, queries2)
    return list(results1), list(results2)

# 실행 및 시간 측정
if __name__ == "__main__":
    # 각 모델에 전달할 쿼리
    queries_for_model1 = [
        "Explain the concept of transformers in AI.",
        "What is the capital of France?",
    ]
    queries_for_model2 = [
        "Summarize the benefits of using Hugging Face models.",
        "Describe the difference between AI and machine learning.",
    ]

    # 시간 측정 시작
    start_time = time.time()

    # 병렬 처리 실행
    results_model1, results_model2 = process_queries_with_two_models(
        queries_for_model1, queries_for_model2
    )

    # 시간 측정 종료
    elapsed_time = time.time() - start_time

    # 결과 출력
    print("Results from Model 1:")
    for i, result in enumerate(results_model1):
        print(f"Query {i+1}: {queries_for_model1[i]}")
        print(f"Result {i+1}: {result}")
        print("-" * 50)

    print("Results from Model 2:")
    for i, result in enumerate(results_model2):
        print(f"Query {i+1}: {queries_for_model2[i]}")
        print(f"Result {i+1}: {result}")
        print("-" * 50)

    # 처리 시간 출력
    print(f"Time taken for parallel processing with two models: {elapsed_time:.2f} seconds")
