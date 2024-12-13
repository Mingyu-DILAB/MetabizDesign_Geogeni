import os
import sys

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from rag_chatbot import RAGChatbot

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def load_documents(file_path):
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=600,
            separator = "\n",
            chunk_overlap=0
        )
        
        return text_splitter.split_documents(documents)
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)

def main():
    try:
        # Initialize the chatbot
        chatbot = RAGChatbot()
        
        # Load and add documents
        docs = load_documents("output2.txt")
        chatbot.add_document(docs)
        
        # Interactive chat loop
        while True:
            try:
                query = input("You: ")
                if query.lower() in ['quit', 'exit', 'bye']:
                    break
                    
                response = chatbot.chat(query)
                print("Assistant:", response)

            except Exception as e:
                print(f"Error processing query: {e}")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()