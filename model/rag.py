#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
from typing import List
import json
import argparse

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

# Configuration
BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / 'settings.json'

# Load settings
with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
    settings = json.load(f)

EMBED_MODEL = settings['embed_model']

def clone_repo(repo_url: str, local_path: Path) -> None:
    if not local_path.exists():
        print(f"Cloning repo {repo_url} into {local_path}...")
        subprocess.run(["git", "clone", repo_url, str(local_path)], check=True)
    else:
        print(f"Repository already exists at {local_path}")


def extract_repo_files(repo_path: Path) -> List[Document]:
    docs: List[Document] = []
    for path in repo_path.rglob('*'):
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
            docs.append(Document(page_content=text, metadata={'source': str(path)}))
        except Exception as e:
            print(f"Warning: could not read {path}: {e}", file=sys.stderr)
    return docs


def build_embeddings_index(
    repo_path: Path,
    index_path: Path,
    embed_model_name: str
) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    raw_docs = extract_repo_files(repo_path)

    chunks: List[Document] = []
    for doc in raw_docs:
        splits = splitter.split_text(doc.page_content)
        for chunk_text in splits:
            chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))

    embedder = SentenceTransformer(embed_model_name)
    class BTEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return embedder.encode(texts, show_progress_bar=True)
        def embed_query(self, text: str) -> List[float]:
            return embedder.encode([text])[0]

    embedding = BTEmbeddings()

    if not index_path.exists():
        print("Building FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embedding)
        vectorstore.save_local(str(index_path))
        print("FAISS index built and saved.")
    else:
        print(f"FAISS index already exists at {index_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embeddings index for a codebase using FAISS and SentenceTransformers.")
    parser.add_argument('--repo-url', type=str, help='Git repository URL to clone')
    parser.add_argument('--output', type=str, help='Output directory for the cloned repo and index')
    args = parser.parse_args()

    repo_url = args.repo_url
    local_repo = Path(args.output, 'repo')
    vector_index_path = Path(args.output, 'faiss_index')

    clone_repo(repo_url, local_repo)
    build_embeddings_index(local_repo, vector_index_path, EMBED_MODEL)


if __name__ == '__main__':
    main()
