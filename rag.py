import os
import sys
import hashlib
import re
import json
import datetime
from typing import List

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path
from backend.config import EMBEDDING_MODEL_NAME, get_embedding_dimension

BASE_DIR = Path(__file__).resolve().parent
BIS_DATA_DIR = BASE_DIR / "data" / "bis"
OUTPUT_INDEX_DIR = BASE_DIR / "faiss_index_bis"

class E5Embeddings(HuggingFaceEmbeddings):
    """
    Custom E5 Embeddings wrapper for SentenceTransformers.
    Prepend 'passage: ' for documents and 'query: ' for user queries.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        formatted = [t if t.startswith("passage: ") else f"passage: {t}" for t in texts]
        return super().embed_documents(formatted)

    def embed_query(self, text: str) -> List[float]:
        formatted = text if text.startswith("query: ") else f"query: {text}"
        return super().embed_query(formatted)

def determine_category(filename, text):
    fn = filename.lower()
    tx = text.lower()
    
    if "hallmark" in fn or "hallmarking" in tx:
        return "hallmarking"
    elif "lab" in fn or "assaying" in fn or "laboratory" in tx or "assaying" in tx:
        return "laboratories"
    elif "regulat" in fn or "gazette" in fn or "order" in fn or "notification" in fn:
        return "regulations"
    elif "licence" in fn or "coc" in fn or "surveillance" in fn or "conformity" in fn or "renewal" in fn or "ca_" in fn or "guideline" in fn:
        return "certification"
    elif "bs_" in fn or "standard" in fn or "indian standard" in tx:
        return "standards"
    elif "recruitment" in fn or "admin" in fn or "bis" in fn:
        return "general_bis"
    else:
        return "other"

def clean_title(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'\s*\(\d+\)$', '', name) # remove duplicate suffix like (1)
    name = name.replace("-", " ").replace("_", " ")
    return name.title()

def extract_identifiers(text):
    is_matches = re.findall(r'\bIS\s*[:/-]?\s*\d+(?:\s*\(Part\s*\d+\))?(?::\s*\d{4})?\b', text, re.IGNORECASE)
    section_matches = re.findall(r'\bsection\s+\d+(?:\(\d+\))?\b', text, re.IGNORECASE)
    clause_matches = re.findall(r'\bclause\s+\d+(?:\.\d+)*\b', text, re.IGNORECASE)
    scheme_matches = re.findall(r'\bScheme\s*[-–]?\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X|\d+)\b', text, re.IGNORECASE)
    qco_matches = re.findall(r'\bQuality\s+Control\s+Order\b|\bQCO\b', text, re.IGNORECASE)
    
    return {
        "is_number": ", ".join(sorted(list(set(is_matches)))) if is_matches else "",
        "section_number": ", ".join(sorted(list(set(section_matches)))) if section_matches else "",
        "clause_number": ", ".join(sorted(list(set(clause_matches)))) if clause_matches else "",
        "scheme_number": ", ".join(sorted(list(set(scheme_matches)))) if scheme_matches else "",
        "qco_reference": "Quality Control Order" if qco_matches else ""
    }

def load_bis_documents():
    documents = []
    failed_files = []
    seen_hashes = {}
    duplicate_files = []
    
    if not os.path.exists(BIS_DATA_DIR):
        print(f"❌ Directory not found: {BIS_DATA_DIR}")
        return documents, failed_files, duplicate_files

    files = [f for f in os.listdir(BIS_DATA_DIR) if f.endswith(".pdf")]
    print(f"📄 Found {len(files)} total PDF files in '{BIS_DATA_DIR}'.")
    
    for filename in sorted(files):
        filepath = os.path.join(BIS_DATA_DIR, filename)
        doc_title = clean_title(filename)
        
        try:
            reader = pypdf.PdfReader(filepath)
            total_pages = len(reader.pages)
            
            full_text = ""
            page_texts = []
            for page_idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                page_texts.append(txt)
                full_text += txt + "\n"
            
            clean_full = re.sub(r'\s+', '', full_text)
            if not clean_full:
                print(f"⚠️ Warning: No extractable text in '{filename}'")
                failed_files.append({"filename": filename, "reason": "No extractable text (scanned or empty)"})
                continue
                
            text_hash = hashlib.md5(clean_full.encode('utf-8')).hexdigest()
            
            if text_hash in seen_hashes:
                original_file = seen_hashes[text_hash]
                print(f"  ⏭️ Skipping DUPLICATE document: '{filename}' (identical text content to '{original_file}')")
                duplicate_files.append({"filename": filename, "duplicate_of": original_file})
                continue
            
            seen_hashes[text_hash] = filename
            
            for page_idx, text in enumerate(page_texts):
                page_num = page_idx + 1
                if text.strip():
                    category = determine_category(filename, text)
                    identifiers = extract_identifiers(text)
                    
                    doc_metadata = {
                        "source_document": filename,
                        "source_path": filepath,
                        "category": category,
                        "page_number": page_num,
                        "document_title": doc_title,
                        "total_pages": total_pages,
                        **identifiers
                    }
                    
                    documents.append(Document(page_content=text, metadata=doc_metadata))
            
            print(f"  ✓ Indexed unique document '{filename}' ({total_pages} pages)")
                
        except Exception as e:
            print(f"❌ Error processing '{filename}': {e}")
            failed_files.append({"filename": filename, "reason": str(e)})

    return documents, failed_files, duplicate_files

def build_bis_index():
    print("🚀 Starting BIS PDF Ingestion & Deduplication Pipeline...")
    raw_docs, failed_files, duplicate_files = load_bis_documents()
    
    print(f"\n📊 DEDUPLICATION REPORT:")
    print(f"  - Unique documents indexed: {len(set(doc.metadata['source_document'] for doc in raw_docs))}")
    print(f"  - Duplicate documents detected & skipped: {len(duplicate_files)}")
    
    if not raw_docs:
        print("❌ No valid documents extracted. Aborting index build.")
        return

    print(f"\n✂️ Splitting {len(raw_docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunked_docs = splitter.split_documents(raw_docs)
    print(f"✅ Generated {len(chunked_docs)} unique document chunks.")

    print(f"🧠 Generating Multilingual Embeddings ({EMBEDDING_MODEL_NAME})...")
    model_kwargs = {'device': 'cpu'}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        model_kwargs['token'] = hf_token

    embeddings = E5Embeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )

    print("💾 Creating FAISS Vector Database...")
    db = FAISS.from_documents(chunked_docs, embeddings)
    
    OUTPUT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(OUTPUT_INDEX_DIR)
    
    dimension = get_embedding_dimension(EMBEDDING_MODEL_NAME)
    meta_info = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "dimension": dimension,
        "total_chunks": len(chunked_docs),
        "built_at": datetime.datetime.now().isoformat()
    }
    
    meta_file = OUTPUT_INDEX_DIR / "index_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"🎉 BIS Vector Database saved successfully to '{OUTPUT_INDEX_DIR}'!")
    print(f"   Model: {EMBEDDING_MODEL_NAME} | Dimension: {dimension} | Chunks: {len(chunked_docs)}")

if __name__ == "__main__":
    build_bis_index()