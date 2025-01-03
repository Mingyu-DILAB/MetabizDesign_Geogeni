import os, sys, argparse
import numpy as np
from rag_chatbot import RAGChatbot
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

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
    documents_path = os.path.join(os.path.dirname(__file__), "dataset", "MinerU", "MyOCR", "results_2", "mineru_output")

    chunks = []
    for root, dirs, files in os.walk(documents_path):
        if "chunks.txt" in files:
            with open(os.path.join(root, "chunks.txt"), "r", encoding="utf-8") as chunk_file:
                lines = chunk_file.readlines()
                for line in lines:
                    chunks.append(line.strip())
            continue

    return chunks

def calculate_f1(true_answers, pred_answers):
    def f1_single(true, pred):
        true_tokens = set(true.split())
        pred_tokens = set(pred.split())
        common_tokens = true_tokens & pred_tokens

        if not common_tokens:
            return 0.0

        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    f1_scores = [f1_single(t, p) for t, p in zip(true_answers, pred_answers)]
    return np.mean(f1_scores)

def calculate_rouge_l(true_answers, pred_answers):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [
        scorer.score(t, p)['rougeL'].fmeasure for t, p in zip(true_answers, pred_answers)
    ]
    return np.mean(rouge_l_scores)

def calculate_bert_score(true_answers, pred_answers, lang="en"):
    P, R, F1 = bert_score(pred_answers, true_answers, lang=lang)
    return np.mean(F1)

def evaluate_performance(chatbot, queries, true_answers, top_k=5):
    """
    주어진 쿼리와 정답에 대해 평가 수행
    :param chatbot: RAGChatbot 객체
    :param queries: 사용자 질문 리스트
    :param true_answers: 정답 리스트
    :param top_k: 생성된 답변 중 상위 선택 개수
    """
    
    pred_answers = []
    for query in queries:
        response = chatbot.generate_response(query, top_k=top_k)
        pred_answers.append(response)


def main(args):
    # Load chunks
    chunks = load_chunks()

    # Initialize the chatbot
    chatbot = RAGChatbot(
        chunks=chunks,
        retriever_model_name=args.retriever_model_name,
        generator_model_name=args.generator_model_name,
    )

    # Build FAISS index
    chatbot.build_index()
    
    queries = [
        "모래의 전단강도에 가장 큰 영향을 미치는 요소는 무엇인가요?",
        "콘크리트의 강도를 결정짓는 주요 인자는 무엇인가요?",
    ]
    true_answers = [
        "모래의 전단강도에 가장 큰 영향을 미치는 요소는 입도와 다짐 상태입니다.",
        "콘크리트의 강도를 결정짓는 주요 인자는 물-시멘트 비와 골재의 품질입니다.",
    ]
    
    evaluate_performance(chatbot, queries, true_answers, top_k=5)
    
    
    
    
    
    

    # User query
    query = "모래의 전단강도에 가장 큰 영향을 미치는 요소는 무엇인가요?"
    report = ""
    response = chatbot.generate_response(query, report, top_k=5)
    print(response)
    

if __name__ == "__main__":
    args = define_argparser()
    main(args)