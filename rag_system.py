"""
RAG System for Cimaprompter - Module 16
Implements Retrieval Augmented Generation using ChromaDB and LangChain

This module provides:
- PDF document loading and processing
- Text chunking with overlap
- Vector embeddings with Google Gemini
- Persistent vector storage with ChromaDB
- Semantic similarity retrieval

Team: Cimaprompter (UABC)
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Complete RAG system for academic QA assistant.

    Handles document loading, embedding, storage, and retrieval
    using ChromaDB as vector store and Google Gemini for embeddings.
    """

    def __init__(
        self,
        documents_path: str,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        embedding_model: str = "huggingface",  # "huggingface" or "gemini"
        google_api_key: Optional[str] = None
    ):
        """
        Initialize the RAG system.

        Args:
            documents_path: Path to directory containing PDF documents
            persist_directory: Where to store ChromaDB data
            chunk_size: Size of text chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
            embedding_model: "huggingface" (local, no API limits) or "gemini" (requires API key)
            google_api_key: Google API key (only needed if embedding_model="gemini")
        """
        self.documents_path = Path(documents_path)
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        # Get API key (only if using Gemini)
        if embedding_model == "gemini":
            self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Set it as environment variable "
                    "or pass it to RAGSystem constructor."
                )
        else:
            self.api_key = None

        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        logger.info(f"RAG System initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get or create Google Gemini embeddings instance."""
        if self.embeddings is None:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",  # Gemini v1beta embedding model
                google_api_key=self.api_key
            )
            logger.info("Google Gemini embeddings initialized")
        return self.embeddings

    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the documents path.

        Returns:
            List of LangChain Document objects
        """
        documents = []
        pdf_files = list(self.documents_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_path}")
            return documents

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            try:
                logger.info(f"Loading: {pdf_path.name}")
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()

                # Add source metadata
                for page in pages:
                    page.metadata['source_file'] = pdf_path.name
                    page.metadata['source_path'] = str(pdf_path)

                documents.extend(pages)
                logger.info(f"  ✓ Loaded {len(pages)} pages from {pdf_path.name}")

            except Exception as e:
                logger.error(f"  ✗ Error loading {pdf_path.name}: {str(e)}")
                continue

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"  ✓ Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document], batch_size: int = 80) -> Chroma:
        """
        Create or load ChromaDB vectorstore with rate limiting for Gemini API.

        Args:
            chunks: List of chunked Document objects
            batch_size: Number of chunks to process per batch (for rate limiting)

        Returns:
            ChromaDB vectorstore
        """
        embeddings = self._get_embeddings()

        # Check if vectorstore already exists
        if Path(self.persist_directory).exists():
            logger.info(f"Loading existing vectorstore from {self.persist_directory}")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
            logger.info("  ✓ Vectorstore loaded")
        else:
            logger.info(f"Creating new vectorstore in {self.persist_directory}")

            # If using Gemini and have many chunks, process in batches with delays
            if self.embedding_model == "gemini" and len(chunks) > batch_size:
                logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size} (rate limiting)...")

                # Create vectorstore with first batch
                first_batch = chunks[:batch_size]
                vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"  ✓ Batch 1/{(len(chunks) + batch_size - 1) // batch_size} processed ({len(first_batch)} chunks)")

                # Process remaining batches with delays
                for i in range(batch_size, len(chunks), batch_size):
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(chunks) + batch_size - 1) // batch_size

                    # Wait 65 seconds between batches (to respect 100 req/min limit)
                    logger.info(f"  ⏳ Waiting 65 seconds for rate limit...")
                    time.sleep(65)

                    batch = chunks[i:i + batch_size]
                    vectorstore.add_documents(batch)
                    logger.info(f"  ✓ Batch {batch_num}/{total_batches} processed ({len(batch)} chunks)")

                logger.info(f"  ✓ Vectorstore created with {len(chunks)} chunks total")
            else:
                # Process all at once (for HuggingFace or small datasets)
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"  ✓ Vectorstore created with {len(chunks)} chunks")

        self.vectorstore = vectorstore
        return vectorstore

    def build_knowledge_base(self) -> Tuple[int, int]:
        """
        Complete workflow: load, chunk, and store documents.

        Returns:
            Tuple of (num_documents, num_chunks)
        """
        logger.info("=" * 60)
        logger.info("Building Knowledge Base")
        logger.info("=" * 60)

        # Load documents
        documents = self.load_documents()
        if not documents:
            logger.error("No documents loaded. Cannot build knowledge base.")
            return 0, 0

        # Split into chunks
        chunks = self.split_documents(documents)

        # Create vectorstore
        self.create_vectorstore(chunks)

        logger.info("=" * 60)
        logger.info(f"✓ Knowledge Base built successfully!")
        logger.info(f"  Documents: {len(documents)}")
        logger.info(f"  Chunks: {len(chunks)}")
        logger.info("=" * 60)

        return len(documents), len(chunks)

    def retrieve(self, query: str, k: int = 3, relevance_threshold: float = 0.6) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query.

        ChromaDB returns cosine distance scores (0 = identical, 2 = opposite).
        Chunks with distance > relevance_threshold are considered irrelevant and discarded.
        Based on Module 16 testing, good hits scored 0.45-0.55; off-topic queries scored higher.

        Args:
            query: User's question
            k: Number of chunks to retrieve before filtering
            relevance_threshold: Max allowed cosine distance (default 0.6). Chunks above
                                  this threshold are discarded as insufficiently relevant.

        Returns:
            List of dicts with 'content', 'source', 'page', 'score'.
            Returns empty list if no chunks pass the relevance threshold.
        """
        if self.vectorstore is None:
            logger.error("Vectorstore not initialized. Call build_knowledge_base() first.")
            return []

        # Perform similarity search with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Format and filter by relevance threshold
        formatted_results = []
        for doc, score in results:
            if float(score) <= relevance_threshold:
                formatted_results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'score': float(score),
                    'metadata': doc.metadata
                })
            else:
                logger.info(f"  Discarded chunk (score {score:.4f} > threshold {relevance_threshold}): "
                            f"{doc.metadata.get('source_file', 'Unknown')} p.{doc.metadata.get('page', '?')}")

        logger.info(f"Retrieved {len(formatted_results)}/{len(results)} chunks "
                    f"(threshold={relevance_threshold}) for query: '{query[:50]}...'")
        return formatted_results

    def get_context_for_llm(self, query: str, k: int = 3) -> str:
        """
        Get formatted context string for LLM prompt.

        Args:
            query: User's question
            k: Number of chunks to retrieve

        Returns:
            Formatted string with retrieved context
        """
        results = self.retrieve(query, k=k)

        if not results:
            return "No relevant information found in the knowledge base."

        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['source']
            page = result['page']
            content = result['content'].strip()

            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{content}\n"
            )

        return "\n---\n\n".join(context_parts)

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        if self.vectorstore is None:
            return {
                'status': 'not_initialized',
                'num_chunks': 0
            }

        # Get collection info
        collection = self.vectorstore._collection
        count = collection.count()

        return {
            'status': 'initialized',
            'num_chunks': count,
            'persist_directory': self.persist_directory,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_rag_system(
    documents_path: str = "./code styles",
    persist_directory: str = "./chroma_db",
    rebuild: bool = False,
    embedding_model: str = "gemini",
    google_api_key: str = None
) -> RAGSystem:
    """
    Initialize or load RAG system.

    Args:
        documents_path: Path to PDF documents
        persist_directory: Where to store ChromaDB
        rebuild: If True, rebuild knowledge base from scratch
        embedding_model: "gemini" or "huggingface"
        google_api_key: Optional API key (overrides env var)

    Returns:
        Initialized RAGSystem instance
    """
    rag = RAGSystem(
        documents_path=documents_path,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        google_api_key=google_api_key
    )

    # Check if we need to build
    if rebuild or not Path(persist_directory).exists():
        logger.info("Building knowledge base...")
        rag.build_knowledge_base()
    else:
        logger.info("Loading existing knowledge base...")
        embeddings = rag._get_embeddings()
        rag.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logger.info("  ✓ Knowledge base loaded")

    return rag


def test_rag_system(rag: RAGSystem, test_queries: Optional[List[str]] = None):
    """
    Test RAG system with sample queries.

    Args:
        rag: Initialized RAGSystem instance
        test_queries: List of test questions (uses defaults if None)
    """
    if test_queries is None:
        test_queries = [
            "¿Cuál es el estándar de indentación en C++?",
            "¿Cómo se nombran las variables en Objective-C?",
            "¿Qué reglas hay para los encabezados en Markdown?"
        ]

    print("\n" + "=" * 60)
    print("Testing RAG System")
    print("=" * 60)

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        results = rag.retrieve(query, k=2)

        for i, result in enumerate(results, 1):
            print(f"\n🔍 Result {i}:")
            print(f"   Source: {result['source']}")
            print(f"   Page: {result['page']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Content preview: {result['content'][:150]}...")

        print("\n" + "-" * 60)


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    rebuild = "--rebuild" in sys.argv

    try:
        # Initialize RAG system
        print("\n🚀 Initializing RAG System...")
        rag_system = initialize_rag_system(
            documents_path="./code styles",
            persist_directory="./chroma_db",
            rebuild=rebuild
        )

        # Show stats
        stats = rag_system.get_stats()
        print(f"\n📊 Knowledge Base Stats:")
        print(f"   Status: {stats['status']}")
        print(f"   Chunks: {stats['num_chunks']}")
        print(f"   Chunk size: {stats['chunk_size']}")
        print(f"   Chunk overlap: {stats['chunk_overlap']}")

        # Run tests
        test_rag_system(rag_system)

        print("\n✅ RAG System test completed successfully!")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
