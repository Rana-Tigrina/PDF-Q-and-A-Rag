import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentEngine:
    def __init__(self):
        self.pdf_dir = "./pdfs"
        self.model_dir = "./model"
        self.db_dir = "./vectorstore"
        
        # Initialize memory without memory_key
        self.memory = ConversationBufferMemory(
            return_messages=True
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            self.llm = LlamaCpp(
                model_path="./model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                temperature=0.75,
                max_tokens=4096,  # Increased from 2000
                n_ctx=8192,       # Added larger context window
                n_batch=512,      # Added batch size
                top_p=1,
                verbose=True     # For debugging
            )
        except Exception as e:
            print(f"Error initializing LlamaCpp: {e}")

    def load_documents(self) -> List:
        """Load and split PDF documents"""
        documents = []
        try:
            for pdf_file in os.listdir(self.pdf_dir):
                if pdf_file.endswith(".pdf"):
                    pdf_path = os.path.join(self.pdf_dir, pdf_file)
                    logger.info(f"Loading PDF: {pdf_path}")
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
            
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        return splits

    def setup_vectorstore(self, documents):
        """Initialize and persist vector store"""
        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
            vectorstore.persist()
            return vectorstore
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    def setup_chain(self, vectorstore):
        """Setup retrieval chain"""
        try:
            # Simplified chain configuration
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                memory_key="chat_history",  # Ensure this matches the default or is explicitly set
                return_source_documents=True,
                verbose=True
                )
            return chain
        except Exception as e:
            logger.error(f"Error setting up chain: {str(e)}")
            raise

    def query(self, question: str):
        """Process user query"""
        try:
            if not hasattr(self, 'chain'):
                docs = self.load_documents()
                vectorstore = self.setup_vectorstore(docs)
                self.chain = self.setup_chain(vectorstore)
            
            result = self.chain({"question": question})
            return result['answer']
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}"
