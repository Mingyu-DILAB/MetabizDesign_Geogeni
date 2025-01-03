import os, sys, argparse
from rag_chatbot import RAGChatbot
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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
    documents_path = os.path.join(os.path.dirname(__file__), "dataset", "MinerU", "MyOCR", "results", "mineru_output")

    chunks = []
    for root, dirs, files in os.walk(documents_path):
        if "chunks.txt" in files:
            with open(os.path.join(root, "chunks.txt"), "r", encoding="utf-8") as chunk_file:
                lines = chunk_file.readlines()
                for line in lines:
                    chunks.append(line.strip())
            continue

    return chunks


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

    # User query
    query = "모래의 전단강도에 가장 큰 영향을 미치는 요소는 무엇인가요?"
    report = ""
    response = chatbot.generate_response(query, report ,top_k=5)
    print(response)
    

if __name__ == "__main__":
    args = define_argparser()
    main(args)