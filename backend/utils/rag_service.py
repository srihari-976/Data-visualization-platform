import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import uuid

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedder = None
        self.initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB and SentenceTransformer"""
        try:
            # Use local persistent storage for ChromaDB
            persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'chroma_db')
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)
                
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="dataset_schemas",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding model (using a lightweight one)
            logger.info("Loading SentenceTransformer model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.initialized = True
            logger.info("RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            self.initialized = False

    def _generate_schema_text(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive text description of the dataframe schema"""
        try:
            # Basic info
            buffer = []
            buffer.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Column details
            buffer.append("\nColumn Details:")
            for col in df.columns:
                dtype = df[col].dtype
                null_count = df[col].isnull().sum()
                sample_vals = df[col].dropna().head(3).tolist()
                buffer.append(f"- {col} ({dtype}): {null_count} missing values. Examples: {sample_vals}")
            
            # Statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                buffer.append("\nNumeric Statistics:")
                stats = df[numeric_cols].describe().to_string()
                buffer.append(stats)
            
            # Categorical summary
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                buffer.append("\nCategorical Summaries:")
                for col in cat_cols:
                    top_vals = df[col].value_counts().head(5).to_dict()
                    buffer.append(f"- {col}: Top values: {top_vals}")
            
            return "\n".join(buffer)
            
        except Exception as e:
            logger.error(f"Error generating schema text: {e}")
            return "Error generating schema summary."

    def _chunk_text(self, text: str, chunk_size: int = 800) -> list[str]:
        """Split text into semantic chunks (at newline boundaries)"""
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_len = 0
        for line in lines:
            if current_len + len(line) > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_len = len(line)
            else:
                current_chunk.append(line)
                current_len += len(line) + 1
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks or [text]

    def add_dataset_schema(self, dataset_id: str, df: pd.DataFrame):
        """Process dataset and store schema chunks in ChromaDB"""
        if not self.initialized:
            logger.warning("RAG Service not initialized, skipping schema storage")
            return

        try:
            logger.info(f"Processing schema for dataset {dataset_id}")
            
            # 1. Generate Schema Text
            full_schema = self._generate_schema_text(df)
            
            # 2. Chunk text
            chunks = self._chunk_text(full_schema)
            
            # 3. Embed chunks
            embeddings = self.embedder.encode(chunks).tolist()
            
            # 4. Prepare metadata and IDs
            metadatas = [{'dataset_id': dataset_id, 'chunk_index': i} for i in range(len(chunks))]
            ids = [f"{dataset_id}_chunk_{i}" for i in range(len(chunks))]
            
            # 5. Add to ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(chunks)} schema chunks for dataset {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error adding dataset schema to RAG: {e}")

    def query_schema(self, dataset_id: str, query: str, n_results: int = 3) -> list[str]:
        """Retrieve relevant schema chunks for a user query"""
        if not self.initialized:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query]).tolist()
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where={"dataset_id": dataset_id}
            )
            
            # Extract documents (chunks)
            if results and results['documents']:
                return results['documents'][0]
            
            return []
            
        except Exception as e:
            logger.error(f"Error querying schema: {e}")
            return []

# Singleton instance
rag_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service
