import os, sys, json, argparse
import torch

from tqdm import tqdm
from typing import List, Optional
from peft import PeftModel, AutoPeftModelForCausalLM
from typing import List, Optional
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

class RAGChatbot:
    def __init__(self, documents_path="./documents"):
        try:
            # Initialize embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr",
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize LLM
            self.model = AutoModelForCausalLM.from_pretrained(
                'MLP-KTLim/llama-3-Korean-Bllossom-8B',
                device_map='auto',
                torch_dtype=torch.float16,
                resume_download=True,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained('MLP-KTLim/llama-3-Korean-Bllossom-8B', padding=True, truncation=True, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(tokenizer))

            # Initialize db with a placeholder
            self.db = None
            
            
        except Exception as e:
            print(f"Error initializing RAGChatbot: {e}")
            raise


    def generate_response(self, user):
        self.model.eval()
        
        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        context = retriever.invoke(user)
        print(context)
        
        messages = [
            {"role": "system", "content": f'''
             한국어로 항상 대답하며, {context}에 맞추어 대답하도록 하되, 주어진 json구조에서 'text'키만 
             참고하세요. 또한 주어진 질문에만 맞게 간결하게 대답하며 굳이 추가적인 정보를
             제공하지 마세요.'''},
            {"role": "user", "content": user}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)



        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def add_document(self, docs: List):
        if not docs:
            print("No documents to add")
            return

        try:
            # If db is None, create a new FAISS store
            if self.db is None:
                self.db = FAISS.from_documents(docs, self.embeddings)
            else:
                # Add new documents to existing vector store
                self.db.add_documents(docs)
            
            print(f"Added {len(docs)} documents to the vector store")
        
        except Exception as e:
            print(f"Error adding documents: {e}")
            
    def chat(self, query: str) -> str:  
        try:
            response = self.generate_response(query)
            return response
        
        except Exception as e:
            print(f"Error during chat: {e}")
            return "I encountered an error while processing your query."
