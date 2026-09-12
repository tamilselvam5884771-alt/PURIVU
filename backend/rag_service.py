import os
import sys
import re
import hashlib
import time
from typing import Dict, Any, List

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path

# Load environment variables
load_dotenv()

# Deployment-safe project base directory
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_BIS_DIR = BASE_DIR / "faiss_index_bis"
FAISS_DEF_DIR = BASE_DIR / "faiss_index"

# Prioritize BIS FAISS vector index specifically
if (FAISS_BIS_DIR / "index.faiss").exists() and (FAISS_BIS_DIR / "index.pkl").exists():
    INDEX_DIR = FAISS_BIS_DIR
elif (FAISS_DEF_DIR / "index.faiss").exists() and (FAISS_DEF_DIR / "index.pkl").exists():
    INDEX_DIR = FAISS_DEF_DIR
else:
    INDEX_DIR = FAISS_BIS_DIR

# Model priority list for fallback handling (Valid Gemini API models)
MODEL_NAMES = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-latest", "gemini-pro-latest"]

class RAGService:
    def __init__(self):
        self.db = None
        self.embeddings = None
        self.initialized = False
        self._init_error = None

    def initialize(self):
        if self.initialized:
            return
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self._init_error = "GEMINI_API_KEY is missing from environment. Set GEMINI_API_KEY in Railway Variables."
            print(f"[RAG_INIT_ERROR] {self._init_error}")
            raise ValueError(self._init_error)

        genai.configure(api_key=api_key)

        index_faiss = INDEX_DIR / "index.faiss"
        index_pkl = INDEX_DIR / "index.pkl"

        print(f"[RAG_DIAG] Resolved BASE_DIR: {BASE_DIR}")
        print(f"[RAG_DIAG] Resolved INDEX_DIR: {INDEX_DIR}")
        print(f"[RAG_DIAG] index.faiss exists: {index_faiss.exists()}")
        print(f"[RAG_DIAG] index.pkl exists: {index_pkl.exists()}")

        if not index_faiss.exists() or not index_pkl.exists():
            self._init_error = f"FAISS index files ('index.faiss', 'index.pkl') not found in '{INDEX_DIR}'."
            print(f"[RAG_INIT_ERROR] {self._init_error}")
            raise FileNotFoundError(self._init_error)

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.db = FAISS.load_local(
                str(INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.initialized = True
            chunks = self.get_chunk_count()
            print(f"[RAG_DIAG] FAISS vector index loaded successfully. Total chunks (ntotal): {chunks}")
            if chunks == 0:
                print(f"[RAG_WARN] FAISS index loaded but ntotal is 0!")
        except Exception as e:
            self._init_error = f"Failed to load FAISS index from {INDEX_DIR}: {str(e)}"
            print(f"[RAG_INIT_ERROR] {self._init_error}")
            raise RuntimeError(self._init_error) from e

    def is_ready(self) -> bool:
        if not self.initialized:
            try:
                self.initialize()
            except Exception as e:
                if not self._init_error:
                    self._init_error = str(e)
                return False
        return self.initialized and self.db is not None

    def get_chunk_count(self) -> int:
        if self.db and hasattr(self.db, "index"):
            return self.db.index.ntotal
        return 0

    def expand_query(self, query: str) -> List[str]:
        expansion_patterns = [
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
        expanded_terms = []
        q_lower = query.lower()
        for pattern, additions in expansion_patterns:
            if re.search(pattern, q_lower):
                expanded_terms.extend(additions)
        return expanded_terms

    def retrieve_hybrid_documents(self, query: str, top_k: int = 6):
        if not self.is_ready():
            raise RuntimeError(self._init_error or "RAG Service is not initialized.")

        # Search base query
        results = self.db.similarity_search_with_score(query, k=top_k)

        # Expand query and search additional terms
        extra_phrases = self.expand_query(query)
        for phrase in extra_phrases:
            extra_results = self.db.similarity_search_with_score(phrase, k=3)
            results.extend(extra_results)

        # Deduplicate chunks by MD5 text hash, keeping lowest L2 distance
        unique_chunks = {}
        for doc, score in results:
            content_hash = hashlib.md5(doc.page_content.strip().encode('utf-8')).hexdigest()
            if content_hash not in unique_chunks or score < unique_chunks[content_hash][1]:
                unique_chunks[content_hash] = (doc, score)

        sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x[1])
        return sorted_chunks[:top_k]

    def _generate_gemini_content(self, prompt: str) -> str:
        last_error = None
        for model_name in MODEL_NAMES:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = e
                time.sleep(1)
                continue
        raise RuntimeError(f"Gemini API request failed across models: {str(last_error)}")

    def calculate_confidence(self, best_score: float, answer_text: str) -> str:
        if "sorry" in answer_text.lower() and "couldn't find" in answer_text.lower():
            return "Limited Evidence"
        elif best_score < 0.75:
            return "High Confidence"
        elif best_score <= 1.15:
            return "Moderate Confidence"
        else:
            return "Limited Evidence"

    def normalize_response_language(self, language: str = None, question: str = "") -> str:
        if not language or not str(language).strip():
            return self.auto_detect_language(question)
        
        clean = str(language).strip().lower()
        if clean == "auto":
            return self.auto_detect_language(question)
        
        if "tamil" in clean or "தமிழ்" in clean or clean in {"ta", "ta-in"}:
            return "Tamil"
        if "hindi" in clean or "हिन्दी" in clean or clean in {"hi", "hi-in"}:
            return "Hindi"
        if "english" in clean or clean in {"en", "en-in"}:
            return "English"
        
        return self.auto_detect_language(question)

    def auto_detect_language(self, question: str = "") -> str:
        if not question:
            return "English"
        if any('\u0b80' <= c <= '\u0bff' for c in question):
            return "Tamil"
        if any('\u0900' <= c <= '\u097f' for c in question):
            return "Hindi"
        return "English"

    def normalize_query_for_retrieval(self, query: str) -> str:
        # If query contains non-ASCII characters (e.g. Tamil or Hindi unicode script)
        if any(ord(char) > 127 for char in query):
            norm_prompt = f"Translate/normalize the following user query into concise English technical search terms for vector database lookup of Indian Standards: '{query}'. Return ONLY the English search terms with no extra commentary."
            try:
                english_terms = self._generate_gemini_content(norm_prompt).strip()
                return f"{query} {english_terms}"
            except Exception:
                return query
        return query

    def process_query(self, question: str, response_language: str = None) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError(self._init_error or "RAG service is not ready.")

        # Determine target response language
        target_lang = self.normalize_response_language(response_language, question)

        # Normalize query for vector lookup if non-English
        retrieval_query = self.normalize_query_for_retrieval(question)
        hybrid_docs = self.retrieve_hybrid_documents(retrieval_query, top_k=6)

        if not hybrid_docs:
            refusal = "Sorry, I couldn't find that information in the provided BIS documents."
            if target_lang == "Tamil":
                refusal = "மன்னித்துக்கொள்ளுங்கள், வழங்கப்பட்ட BIS ஆவணங்களில் இந்தத் தகவல் கிடைக்கவில்லை. தவறான BIS தரநிலை அல்லது certification requirement-ஐ உருவாக்காமல் இருக்க, நான் இதை உறுதிப்படுத்தப்பட்ட தகவலாகக் கூறவில்லை."
            elif target_lang == "Hindi":
                refusal = "क्षमा करें, दिए गए BIS दस्तावेजों में यह जानकारी नहीं मिल सकी। किसी गलत जानकारी से बचने के लिए, केवल आधिकारिक दस्तावेजों के आधार पर ही उत्तर दिया जाता है।"
            return {
                "answer": refusal,
                "evidence_status": "Limited Evidence",
                "sources": [],
                "response_language": target_lang
            }

        best_l2_score = hybrid_docs[0][1]

        context_chunks = []
        sources_list = []
        seen_sources = set()

        for i, (doc, score) in enumerate(hybrid_docs):
            src_doc = doc.metadata.get("source_document", "BIS Document")
            doc_title = doc.metadata.get("document_title", src_doc)
            page_num = doc.metadata.get("page_number", "N/A")
            cat = doc.metadata.get("category", "general")
            is_num = doc.metadata.get("is_number", "")
            clause_num = doc.metadata.get("clause_number", "")
            sec_num = doc.metadata.get("section_number", "")
            scheme_num = doc.metadata.get("scheme_number", "")
            qco_ref = doc.metadata.get("qco_reference", "")

            meta_str = f"Document: {doc_title} ({src_doc}) | Category: {cat} | Page: {page_num}"
            if is_num: meta_str += f" | IS: {is_num}"
            if clause_num: meta_str += f" | Clause: {clause_num}"
            if sec_num: meta_str += f" | Section: {sec_num}"
            if scheme_num: meta_str += f" | Scheme: {scheme_num}"
            if qco_ref: meta_str += f" | QCO Ref: {qco_ref}"

            context_chunks.append(
                f"SOURCE {i+1}\n"
                f"{meta_str}\n"
                f"Relevant Text:\n{doc.page_content}"
            )

            src_key = f"{src_doc}_p{page_num}"
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                src_item = {
                    "document": src_doc,
                    "document_title": doc_title if doc_title != src_doc else None,
                    "page": page_num,
                    "category": cat if cat != "general" else None,
                    "is_number": is_num if is_num else None,
                    "clause_number": clause_num if clause_num else None,
                    "section_number": sec_num if sec_num else None,
                    "scheme_number": scheme_num if scheme_num else None,
                    "qco_reference": qco_ref if qco_ref else None,
                    "relevance_score": round(float(score), 4)
                }
                sources_list.append(src_item)

        context_text = "\n\n".join(context_chunks)

        if target_lang == "Tamil":
            lang_directive = (
                "CRITICAL MANDATORY LANGUAGE INSTRUCTION:\n"
                "You MUST write your ENTIRE explanation and markdown section headers in TAMIL (தமிழ்).\n"
                "Do NOT write in English. Do NOT switch to English because the Context text is in English.\n"
                "Write natural, fluent, professional Tamil for all explanations.\n\n"
                "TECHNICAL IDENTIFIER RULE (MUST REMAIN IN ENGLISH):\n"
                "Keep ONLY the following technical identifiers in original English script:\n"
                "• BIS\n"
                "• IS numbers (e.g., IS 15820, IS 12312)\n"
                "• QCO (Quality Control Order)\n"
                "• CoC (Certificate of Conformity)\n"
                "• Scheme-I, Scheme-IV, Scheme numbers\n"
                "• Clause numbers (e.g. Clause 5.1)\n"
                "• Section numbers (e.g. Section 16)\n"
                "• Document names and citations\n"
                "Do NOT translate standard numbers or IS references into Tamil script."
            )
        elif target_lang == "Hindi":
            lang_directive = (
                "CRITICAL MANDATORY LANGUAGE INSTRUCTION:\n"
                "You MUST write your ENTIRE explanation and markdown section headers in HINDI (हिन्दी).\n"
                "Do NOT write in English. Do NOT switch to English because the Context text is in English.\n"
                "Write natural, fluent, professional Hindi for all explanations.\n\n"
                "TECHNICAL IDENTIFIER RULE (MUST REMAIN IN ENGLISH):\n"
                "Keep ONLY BIS technical identifiers (BIS, IS 15820, QCO, CoC, Scheme-I, Scheme-IV, Clause numbers, Section numbers) in original English script."
            )
        else:
            lang_directive = (
                "CRITICAL MANDATORY LANGUAGE INSTRUCTION:\n"
                "You MUST write your entire response in English."
            )

        prompt = f"""
You are PURIVU (BIS Saathi), an AI-powered Intelligent Assistant for Indian Standards and Bureau of Indian Standards (BIS) services.

{lang_directive}

CRITICAL GROUNDING RULES:
1. Answer ONLY using the facts explicitly present in the provided Context below.
2. DO NOT hallucinate or invent any Indian Standard (IS) numbers, clause numbers, certification requirements, testing procedures, fees, or legal mandates not stated in the Context.
3. Clearly distinguish between distinct concepts (BIS Certification vs Licence vs CoC vs QCO).
4. If the provided Context does NOT contain enough information to answer the question accurately, respond strictly with refusal text in {target_lang}.
5. Format your response into clean markdown sections in {target_lang}:
   ### Answer
   [Clear, direct explanation in {target_lang}]

   ### Key Points
   • [Point 1 in {target_lang}]
   • [Point 2 in {target_lang}]

   ### What You May Need to Do
   1. [Step 1 in {target_lang}]
   2. [Step 2 in {target_lang}]
   *(NOTE: Omit 'What You May Need to Do' if evidence does not provide actionable steps.)*

Context:
{context_text}

Question:
{question}

Final Answer (MUST BE IN {target_lang}):
"""

        answer_text = self._generate_gemini_content(prompt)
        confidence_level = self.calculate_confidence(best_l2_score, answer_text)

        if "sorry" in answer_text.lower() and "couldn't find" in answer_text.lower():
            confidence_level = "Limited Evidence"

        return {
            "answer": answer_text,
            "evidence_status": confidence_level,
            "sources": sources_list,
            "response_language": target_lang
        }

    def process_vision_image(self, image_bytes: bytes, user_question: str = None) -> Dict[str, Any]:
        import io
        import json
        from PIL import Image

        if not self.is_ready():
            raise RuntimeError(self._init_error or "RAG Service is not ready.")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Invalid image file format: {str(e)}")

        vision_prompt = """
You are an expert product recognition vision system. Analyze the provided product image carefully.
Identify ONLY the physical product details.

CRITICAL RULES:
1. DO NOT state whether BIS certification, Quality Control Orders, or Indian Standards are mandatory.
2. DO NOT invent IS numbers, clause numbers, or legal regulations.
3. Return STRICTLY a single raw JSON object (with no extra markdown code block wrappers) formatted exactly as:
{
  "product_name": "<Specific name, e.g. Electric Kettle, Pressure Cooker, Gold Jewellery, Safety Helmet>",
  "product_category": "<General category, e.g. Household Electrical Appliance, Jewellery, Personal Protective Equipment>",
  "description": "<Brief physical description of visible product features>",
  "confidence": "<High, Moderate, or Low>"
}
"""

        last_error = None
        vision_json = None

        for model_name in MODEL_NAMES:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([vision_prompt, pil_image])
                if response and response.text:
                    clean_text = response.text.strip()
                    # Strip any markdown code fences if present
                    clean_text = re.sub(r'^```(json)?', '', clean_text, flags=re.IGNORECASE).strip()
                    clean_text = re.sub(r'```$', '', clean_text).strip()
                    vision_json = json.loads(clean_text)
                    break
            except Exception as e:
                last_error = e
                time.sleep(1)
                continue

        if not vision_json:
            # Fallback product identification if Vision API JSON parsing fails
            vision_json = {
                "product_name": "Unspecified Product",
                "product_category": "General Goods",
                "description": "Product detected from user image.",
                "confidence": "Low"
            }

        product_info = {
            "name": vision_json.get("product_name", "Unspecified Product"),
            "category": vision_json.get("product_category", "General Goods"),
            "description": vision_json.get("description", "Product detected from image."),
            "confidence": vision_json.get("confidence", "Moderate")
        }

        # Build RAG query using product identification + optional user question
        rag_query = f"{product_info['name']} {product_info['category']}"
        if user_question and user_question.strip():
            rag_query += f" {user_question.strip()}"

        # Run RAG query against local BIS FAISS index
        rag_result = self.process_query(rag_query)

        # Prefix answer to explicitly distinguish Vision Observation from BIS RAG Evidence
        observation_prefix = (
            f"**VISION OBSERVATION**:\n"
            f"Based on the image, the product appears to be **{product_info['name']}** ({product_info['category']}).\n\n"
        )
        
        grounded_answer = observation_prefix + rag_result["answer"]

        return {
            "product": product_info,
            "answer": grounded_answer,
            "evidence_status": rag_result["evidence_status"],
            "sources": rag_result["sources"]
        }


# Global singleton instance
rag_service = RAGService()
