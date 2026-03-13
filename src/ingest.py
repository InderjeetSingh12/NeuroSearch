from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class IngestionEngine:
    """
    Handles the loading and splitting of documents for ingestion.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

    def load_documents(self, source_dir: str, glob_pattern: str = "**/*.txt") -> List[Document]:
        """
        Loads documents from a directory matching a glob pattern.
        """
        print(f"Loading documents from {source_dir} matching '{glob_pattern}'...")
        loader = DirectoryLoader(source_dir, glob=glob_pattern, loader_cls=TextLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits loaded documents into smaller chunks for embedding.
        """
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks
