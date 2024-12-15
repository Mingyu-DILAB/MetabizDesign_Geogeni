import os, sys, json, random
import numpy as np

from typing import List
from FlagEmbedding import BGEM3FlagModel

random.seed(42)

def main():
    # 1. 데이터 준비
    with open(os.path.join("..", "cqa.json"), "r", encoding="utf-8") as json_file:
        data_list = json.load(json_file)

    # 2. 네거티브 샘플링
    # retriever 학습 데이터 포멧에 맞게 {query, pos, neg} 포멧으로 변환
    new_data_list = []
    for i, data in enumerate(data_list):
        pos_chunk = data["Chunk"]

        other_chunks = [item["Chunk"] for j, item in enumerate(data_list) if j != i]
        neg_chunks = random.sample(other_chunks, 10)

        new_data = {"query": data["Question"], "pos": pos_chunk, "neg": neg_chunks}
        new_data_list.append(new_data)

    # 3. 파인튜닝
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
                "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=500)['dense_vecs']
    embeddings_2 = model.encode(sentences_2)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)

if __name__ == "__main__":
    main()