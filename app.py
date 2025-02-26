import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
    
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom embeddings class for Stella model
class StellaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "dunzhang/stella_en_400M_v5"):
        """Initialize the Stella embeddings model for CPU usage"""
        try:
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device="cpu",
                config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
            )
            logger.info(f"Successfully initialized Stella embeddings model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Stella embeddings: {e}")
            raise
        
        self.query_prompt_name = "s2p_query"  # For sentence-to-passage retrieval
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for query with special prompt"""
        try:
            embedding = self.model.encode([text], prompt_name=self.query_prompt_name)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

class ContentEngine:
    def __init__(self):
        self.pdf_dir = "./pdfs"
        self.model_dir = "./model"
        self.db_dir = "./vectorstore"
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Use custom Stella embeddings instead of HuggingFaceBgeEmbeddings
        self.embeddings = StellaEmbeddings(model_name="dunzhang/stella_en_400M_v5")
        
        model_filename = "Mistral-7b-Instruct-v0.3.Q4_K_M.gguf"
        model_path = os.path.join(self.model_dir, model_filename)
        
        # Rest of the initialization remains the same
        if not os.path.isfile(model_path):
            err_msg = f"Model file not found at path: {model_path}. Please ensure the model file is placed in the correct directory."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.25,
                max_tokens=4096,
                n_ctx=8192,
                n_batch=512,
                top_p=1,
                verbose=True
            )
        except Exception as e:
            logger.error(f"Error initializing LlamaCpp: {e}")
            raise

        self.chain = None

    def load_documents(self) -> List:
        """Load and split PDF documents"""
        documents = []
        if not os.path.exists(self.pdf_dir):
            logger.error(f"PDF directory not found: {self.pdf_dir}")
            return documents
        
        for pdf_file in sorted(os.listdir(self.pdf_dir)):
            if pdf_file.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, pdf_file)
                logger.info(f"Loading PDF: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    if docs:
                        documents.extend(docs)
                    else:
                        logger.warning(f"No content loaded from {pdf_path}")
                except Exception as e:
                    logger.error(f"Error loading {pdf_path}: {e}")
        
        if not documents:
            logger.warning("No documents loaded from PDFs.")
            return documents

        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
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
            logger.error(f"Error setting up vector store: {e}")
            raise

    def setup_chain(self, vectorstore):
        """Setup retrieval chain"""
        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                memory_key="chat_history",
                return_source_documents=True,
                verbose=True
            )
            return chain
        except Exception as e:
            logger.error(f"Error setting up chain: {e}")
            raise

    def query(self, question: str):
        """Process user query using cached chain if available"""
        try:
            if self.chain is None:
                documents = self.load_documents()
                if not documents:
                    return "No documents found. Please add PDF files to the pdfs directory."
                vectorstore = self.setup_vectorstore(documents)
                self.chain = self.setup_chain(vectorstore)
            
            result = self.chain({"question": question})
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error: {e}"