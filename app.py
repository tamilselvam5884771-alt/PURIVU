import os
import sys
import re
import hashlib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is not set. Please set it in your .env file.")
    st.stop()

DEFAULT_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite").strip()
MODEL_NAMES = list(dict.fromkeys([DEFAULT_TEXT_MODEL, "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-3.5-flash"]))

def generate_with_fallback(prompt):
    last_err = None
    for mname in MODEL_NAMES:
        try:
            m = genai.GenerativeModel(mname)
            res = m.generate_content(prompt)
            if res and res.text:
                return res.text
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("All Gemini models failed to respond.")


INDEX_DIR = "faiss_index_bis" if os.path.exists("faiss_index_bis") else "faiss_index"

# -----------------------------
# Load FAISS Database
# -----------------------------
@st.cache_resource
def load_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

try:
    db = load_database()
except Exception as e:
    st.error(f"Error loading database from '{INDEX_DIR}':\n\n{e}\n\nPlease run `python rag.py` to build the BIS FAISS index.")
    st.stop()

# -----------------------------
# Query Expansion Rules
# -----------------------------
EXPANSION_PATTERNS = [
    (r'\b(apply|apply for|get|obtain|how to get|procedure for)\b.*\b(certification|approval|licence|license)\b',
     ["Grant of Licence guidelines application procedure Scheme-I", "Grant of CoC guidelines application Scheme-IV", "grant of licence to manufacturer"]),
    (r'\b(quality control order|qco|mandatory|compulsory|is bis mandatory|is certification mandatory)\b',
     ["Quality Control Order QCO section 16 compulsory use of Standard Mark", "compulsory certification order notified goods Scheme-I Scheme-IV"]),
    (r'\b(certificate of conformity|coc)\b',
     ["Certificate of Conformity CoC Scheme-IV grant of CoC conditions of CoC"]),
    (r'\b(hallmark|hallmarking|assaying|ahc)\b',
     ["hallmarking Assaying and Hallmarking Centre A&HC gold jewellery artefacts fineness"]),
    (r'\b(factory surveillance|surveillance)\b',
     ["factory surveillance surveillance inspection review of test reports licensee premises"]),
    (r'\b(non-conformity|non conformity|failure|failure of sample)\b',
     ["dealing with non-conformity failure of sample stop marking corrective actions"])
]

def expand_query(query):
    expanded_terms = []
    q_lower = query.lower()
    for pattern, additions in EXPANSION_PATTERNS:
        if re.search(pattern, q_lower):
            expanded_terms.extend(additions)
    return expanded_terms

def retrieve_hybrid_documents(query, db, top_k=6):
    # Search original query with score
    results = db.similarity_search_with_score(query, k=top_k)
    
    # Expand query if applicable
    extra_phrases = expand_query(query)
    for phrase in extra_phrases:
        extra_results = db.similarity_search_with_score(phrase, k=3)
        results.extend(extra_results)
        
    # Deduplicate results by chunk content hash, keeping best (lowest L2) score
    unique_chunks = {}
    for doc, score in results:
        content_hash = hashlib.md5(doc.page_content.strip().encode('utf-8')).hexdigest()
        if content_hash not in unique_chunks or score < unique_chunks[content_hash][1]:
            unique_chunks[content_hash] = (doc, score)
            
    # Sort by L2 score ascending (lower score = higher similarity)
    sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x[1])
    return sorted_chunks[:top_k]

def calculate_confidence(best_score, answer_text):
    if "sorry" in answer_text.lower() and "couldn't find" in answer_text.lower():
        return "Limited Evidence"
    elif best_score < 0.75:
        return "High Confidence"
    elif best_score <= 1.15:
        return "Moderate Confidence"
    else:
        return "Limited Evidence"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="PURIVU - BIS Saathi Assistant",
    page_icon="🇮🇳",
    layout="centered"
)

st.title("🇮🇳 PURIVU - BIS Saathi")
st.caption("AI-Powered Intelligent Assistant for Indian Standards and BIS Services")
st.markdown(f"**Active Knowledge Database**: `{INDEX_DIR}` *(Deduplicated & Query Expanded)*")
st.write("Ask questions about Indian Standards, BIS schemes, hallmarking, certification guidelines, or Quality Control Orders (QCOs).")

question = st.text_input("Enter your question", placeholder="e.g., How can a manufacturer apply for BIS certification? or What is a Quality Control Order?")

if st.button("Ask", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching official BIS knowledge base..."):
        hybrid_docs = retrieve_hybrid_documents(question, db, top_k=6)

        if not hybrid_docs:
            st.warning("No relevant information found in the BIS knowledge base.")
            st.stop()

        best_l2_score = hybrid_docs[0][1]

        context_chunks = []
        sources_set = set()

        for i, (doc, score) in enumerate(hybrid_docs):
            src_doc = doc.metadata.get("source_document", "BIS Document")
            page_num = doc.metadata.get("page_number", "N/A")
            cat = doc.metadata.get("category", "general")
            doc_title = doc.metadata.get("document_title", src_doc)
            is_num = doc.metadata.get("is_number", "")
            clause_num = doc.metadata.get("clause_number", "")
            scheme_num = doc.metadata.get("scheme_number", "")

            meta_str = f"Document: {doc_title} ({src_doc}) | Category: {cat} | Page: {page_num}"
            if is_num: meta_str += f" | IS: {is_num}"
            if clause_num: meta_str += f" | Clause: {clause_num}"
            if scheme_num: meta_str += f" | Scheme: {scheme_num}"

            context_chunks.append(
                f"SOURCE {i+1}\n"
                f"{meta_str}\n"
                f"Relevant Text:\n{doc.page_content}"
            )
            
            sources_set.add(f"• {doc_title} ({src_doc}) — Page {page_num}")

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
You are PURIVU (BIS Saathi), an AI-powered Intelligent Assistant for Indian Standards and Bureau of Indian Standards (BIS) services.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the facts explicitly present in the provided Context below.
2. DO NOT hallucinate or invent any Indian Standard (IS) numbers, clause numbers, certification requirements, testing procedures, fees, or legal mandates not stated in the Context.
3. Clearly distinguish between distinct concepts:
   - Indian Standards
   - BIS Certification vs BIS Licence vs Certificate of Conformity (CoC)
   - Compulsory Certification / Quality Control Orders (QCOs) vs Voluntary Standards
   - Hallmarking vs General Certification
   - Testing / Laboratory Circulars vs Regulations
4. If the provided Context does NOT contain enough information to answer the question accurately, respond strictly with:
   "Sorry, I couldn't find that information in the provided BIS documents."
5. Format your response into clean markdown sections:
   ### Answer
   [Clear, direct explanation]

   ### Key Points
   • [Point 1]
   • [Point 2]

   ### What You May Need to Do
   1. [Step 1]
   2. [Step 2]
   *(NOTE: Omit 'What You May Need to Do' if the evidence does NOT provide actionable steps.)*

6. AT THE END OF YOUR ANSWER, include a dedicated "Sources:" section listing unique source documents and page numbers used.

Context:
{context_text}

Question:
{question}

Answer:
"""

        try:
            answer_text = generate_with_fallback(prompt)


            confidence_level = calculate_confidence(best_l2_score, answer_text)

            st.subheader("Answer")
            st.markdown(answer_text)

            st.markdown(f"**Evidence Confidence**: `{confidence_level}` *(Relevance Score: {best_l2_score:.4f})*")

            with st.expander("🔍 View Retrieved Sources & Evidence Diagnostics"):
                for idx, (doc, score) in enumerate(hybrid_docs):
                    src_doc = doc.metadata.get("source_document")
                    pg = doc.metadata.get("page_number")
                    cat = doc.metadata.get("category")
                    st.markdown(f"**Source {idx+1}** (L2 Dist: `{score:.4f}`) | `{src_doc}` | Page {pg} | Category: `{cat}`")
                    st.text(doc.page_content[:300] + "...")

        except Exception as e:
            st.error(f"Gemini Error:\n\n{e}")