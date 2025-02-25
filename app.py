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
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            self.llm = LlamaCpp(
                model_path=os.path.join(self.model_dir, "mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
                temperature=0.75,
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