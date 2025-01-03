import os, sys, json, argparse
import torch
import numpy as np

from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM


class RAGChatbot:
    def __init__(self, chunks, retriever_model_name, generator_model_name):
        self.chunks = chunks
        self.retriever_model_name = retriever_model_name
        self.generator_model_name = generator_model_name

        self.vector_store = None
        self.embedding_model = None


    def build_index(self, store_name=os.path.join(os.path.dirname(__file__), "retriever", "faiss_index")):
        """청크 임베딩 저장"""

        if not os.path.exists(os.path.join(os.path.dirname(__file__), "retriever")):
            os.makedirs(os.path.join(os.path.dirname(__file__), "retriever"))

        if self.retriever_model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
            from langchain_openai import OpenAIEmbeddings
            from langchain.storage import LocalFileStore
            from langchain_community.vectorstores import FAISS
            from langchain.embeddings import CacheBackedEmbeddings

            load_dotenv()
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

            # 임베딩 생성 (캐시 적용)
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            store = LocalFileStore(os.path.join(os.path.dirname(__file__), "retriever", "cache"))
            cached_embedder = CacheBackedEmbeddings.from_bytes_store(self.embedding_model, store, namespace=self.embedding_model.model)

            # FAISS 인덱스 생성
            vector_store = FAISS.from_texts(self.chunks, cached_embedder)
            vector_store.save_local(store_name)
            self.vector_store = FAISS.load_local(store_name, self.embedding_model, allow_dangerous_deserialization=True)
            print(f"FAISS index built with {self.vector_store.index.ntotal} chunks. => 'qa_llm/{store_name}'")

        elif self.retriever_model_name == "BAAI/bge-m3":
            from FlagEmbedding import BGEM3FlagModel, FlagReranker

            # 임베딩 생성
            embedding_model_path = os.path.join(os.path.dirname(__file__), "retriever", "bge-m3")
            self.embedding_model = BGEM3FlagModel(
                'BAAI/bge-m3',
                device_map="auto",
                use_fp16=True,
                cache_dir=embedding_model_path,
            )
            
            # 로컬 저장 (FAISS 미지원 모델)
            embedding_save_path = os.path.join(os.path.dirname(__file__), "retriever", "bge-m3_embeddings.npz")
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
            chunks_embedding = np.load(os.path.join(os.path.dirname(__file__), "retriever", "bge-m3_embeddings.npz"))["embeddings"]

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
        generator_model_path = os.path.join(os.path.dirname(__file__), "generator", "MLP-KTLim", "llama-3-Korean-Bllossom-8B")
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