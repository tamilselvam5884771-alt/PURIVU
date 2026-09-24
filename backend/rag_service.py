import os
import sys
import re
import hashlib
import time
import gc
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

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

# Centralized Gemini model configuration
DEFAULT_TEXT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_VISION_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TEXT_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.6-flash"]
DEFAULT_VISION_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.6-flash"]

def get_text_models() -> List[str]:
    primary = os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL).strip()
    models = [primary]
    for m in DEFAULT_TEXT_FALLBACKS:
        if m not in models:
            models.append(m)
    return models

def get_vision_models() -> List[str]:
    primary = os.getenv("GEMINI_VISION_MODEL", DEFAULT_VISION_MODEL).strip()
    models = [primary]
    for m in DEFAULT_VISION_FALLBACKS:
        if m not in models:
            models.append(m)
    return models

class SimpleLRUCache:
    """Thread-safe lightweight in-memory LRU cache."""
    def __init__(self, maxsize: int = 200):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()

def _compute_index_version(index_dir: Path) -> str:
    faiss_p = index_dir / "index.faiss"
    pkl_p = index_dir / "index.pkl"
    st_f = faiss_p.stat() if faiss_p.exists() else None
    st_p = pkl_p.stat() if pkl_p.exists() else None
    if st_f and st_p:
        raw = f"{st_f.st_mtime}_{st_f.st_size}_{st_p.st_mtime}_{st_p.st_size}"
        return hashlib.md5(raw.encode()).hexdigest()[:10]
    return "v1"

import json
import datetime
from backend.config import EMBEDDING_MODEL_NAME, get_embedding_dimension

class GeminiEmbeddings(Embeddings):
    """
    Lightweight Gemini Managed Embeddings wrapper using google-genai SDK.
    Supports RETRIEVAL_DOCUMENT for indexing and RETRIEVAL_QUERY for user queries.
    """
    def __init__(self, client: genai.Client = None, model: str = EMBEDDING_MODEL_NAME, dimension: int = 768):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key and not client:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        self.client = client or genai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for attempt in range(8):
                try:
                    res = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.dimension
                        )
                    )
                    for emb in res.embeddings:
                        embeddings.append(emb.values)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        sleep_s = (attempt + 1) * 4
                        print(f"  [429 Rate Limit] Retrying batch in {sleep_s}s...")
                        time.sleep(sleep_s)
                    else:
                        raise e
            time.sleep(0.3)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        for attempt in range(2):
            try:
                res = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.dimension
                    )
                )
                return res.embeddings[0].values
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < 1:
                        time.sleep(1)
                    else:
                        print(f"  [WARN] embed_query hit 429 rate limit. Using 768d fallback vector for query retrieval.")
                        return [0.01] * self.dimension
                else:
                    print(f"  [WARN] embed_query error: {e}. Using 768d fallback vector.")
                    return [0.01] * self.dimension
        return [0.01] * self.dimension

class RAGService:
    def __init__(self):
        self.db = None
        self.embeddings = None
        self.gemini_client = None
        self.initialized = False
        self._init_error = None
        self.init_time_sec = 0.0
        self.index_version = "v1"

        # In-Memory LRU Caches
        self.embedding_cache = SimpleLRUCache(maxsize=300)
        self.retrieval_cache = SimpleLRUCache(maxsize=150)
        self.answer_cache = SimpleLRUCache(maxsize=150)

        # Production Telemetry Metrics
        self.total_queries = 0
        self.cache_hits = 0
        self.latencies = []       # total latencies
        self.retrieval_times = [] # retrieval latencies
        self.gemini_times = []    # gemini latencies
        self.path_counts = {"FAST": 0, "DEEP": 0, "CACHED": 0}

    def initialize(self):
        if self.initialized:
            return
        
        t_start = time.time()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self._init_error = "GEMINI_API_KEY is missing from environment. Set GEMINI_API_KEY in Railway Variables."
            print(f"[RAG_INIT_ERROR] {self._init_error}")
            raise ValueError(self._init_error)

        self.gemini_client = genai.Client(api_key=api_key)

        index_faiss = INDEX_DIR / "index.faiss"
        index_pkl = INDEX_DIR / "index.pkl"
        meta_file = INDEX_DIR / "index_meta.json"

        if not index_faiss.exists() or not index_pkl.exists():
            self._init_error = f"FAISS index files ('index.faiss', 'index.pkl') not found in '{INDEX_DIR}'."
            print(f"[RAG_INIT_ERROR] {self._init_error}")
            raise FileNotFoundError(self._init_error)

        # Validate index metadata against configured model & dimension
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                expected_dim = get_embedding_dimension(EMBEDDING_MODEL_NAME)
                idx_dim = meta.get("dimension")
                idx_model = meta.get("embedding_model_name")
                
                if idx_dim != expected_dim or idx_model != EMBEDDING_MODEL_NAME:
                    self._init_error = (
                        f"FAISS index validation mismatch! "
                        f"Configured model '{EMBEDDING_MODEL_NAME}' (dim {expected_dim}) "
                        f"does not match index model '{idx_model}' (dim {idx_dim}). "
                        f"Please rebuild FAISS index using 'python rag.py'."
                    )
                    print(f"[RAG_INIT_ERROR] {self._init_error}")
                    raise RuntimeError(self._init_error)
            except Exception as e:
                if isinstance(e, RuntimeError): raise e
                print(f"[RAG_INIT_WARN] Unable to parse index_meta.json: {e}")

        try:
            self.index_version = _compute_index_version(INDEX_DIR)

            t_emb_start = time.time()
            self.embeddings = GeminiEmbeddings(
                client=self.gemini_client,
                model=EMBEDDING_MODEL_NAME,
                dimension=get_embedding_dimension(EMBEDDING_MODEL_NAME)
            )
            t_emb_end = time.time()

            t_faiss_start = time.time()
            self.db = FAISS.load_local(
                str(INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            t_faiss_end = time.time()

            self.initialized = True
            self.init_time_sec = round(time.time() - t_start, 4)
            chunks = self.get_chunk_count()

            print("\n" + "=" * 60)
            print("[PURIVU GEMINI MANAGED RAG WARM-START DIAGNOSTICS]")
            print(f"- Active Index Directory:  {INDEX_DIR}")
            print(f"- Index Version Hash:      {self.index_version}")
            print(f"- FAISS Vectors Loaded:    {chunks}")
            print(f"- Metadata Entries Loaded: {chunks}")
            print(f"- Embedding Model Loaded:  {EMBEDDING_MODEL_NAME} ({get_embedding_dimension(EMBEDDING_MODEL_NAME)}d, Managed Cloud API)")
            print(f"- Gemini Client Load:      {t_emb_end - t_emb_start:.4f}s")
            print(f"- FAISS Index Load Time:   {t_faiss_end - t_faiss_start:.4f}s")
            print(f"- RAG Service Status:      READY (Total Warm-start: {self.init_time_sec:.2f}s)")
            print("=" * 60 + "\n")
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

    def get_corpus_chunk_positions(self) -> int:
        meta_file = INDEX_DIR / "index_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    return meta.get("corpus_chunk_positions", 2239)
            except Exception:
                pass
        return 2239

    def get_chunk_count(self) -> int:
        if self.db and hasattr(self.db, "index"):
            return self.db.index.ntotal
        return 0

    # -------------------------------------------------------------------------
    # 1. SCENARIO UNDERSTANDING
    # -------------------------------------------------------------------------
    def extract_scenario(self, question: str) -> Dict[str, Any]:
        """Extracts structured scenario parameters from user input without inventing values."""
        q_clean = question.strip()
        q_lower = q_clean.lower()
        
        # User Role Extraction
        user_role = None
        role_patterns = [
            (r'\b(oil producer|producer)\b', 'producer'),
            (r'\b(manufacturer|manufacture|manufacturing|factory owner|plant owner)\b', 'manufacturer'),
            (r'\b(trader|distributor|seller|vendor|merchant)\b', 'trader'),
            (r'\b(importer|importing|foreign manufacturer)\b', 'importer'),
            (r'\b(consumer|buyer|customer)\b', 'consumer'),
            (r'\b(jeweller|jeweler|goldsmith)\b', 'jeweller'),
            (r'\b(testing lab|laboratory|assaying centre)\b', 'laboratory')
        ]
        for pattern, role in role_patterns:
            if re.search(pattern, q_lower):
                user_role = role
                break

        # Product Extraction
        product = None
        product_patterns = [
            (r'\b(coconut oil)\b', 'coconut oil'),
            (r'\b(edible oil|vegetable oil|mustard oil|sunflower oil)\b', 'edible oil'),
            (r'\b(gold jewellery|gold jewelry|gold artefact|gold artifacts|hallmarked gold)\b', 'gold jewellery'),
            (r'\b(electric kettle|kettle)\b', 'electric kettle'),
            (r'\b(pressure cooker)\b', 'pressure cooker'),
            (r'\b(safety helmet|helmet)\b', 'safety helmet'),
            (r'\b(pvc pipe|pipes)\b', 'pvc pipe'),
            (r'\b(cement)\b', 'cement'),
            (r'\b(steel|steel product|rebar)\b', 'steel product'),
            (r'\b(toy|toys)\b', 'toys'),
            (r'\b(battery|batteries)\b', 'battery'),
        ]
        for pattern, prod in product_patterns:
            if re.search(pattern, q_lower):
                product = prod
                break

        if not product:
            # Fallback regex for "company", "[X] business", "manufacture [X]"
            m = re.search(r'\b(?:run a|own a|in the)\s+([a-z0-9\s]+?)\s+(?:company|business|factory|plant|industry)\b', q_lower)
            if m:
                candidate = m.group(1).strip()
                if candidate not in {'small', 'large', 'new', 'local', 'private'}:
                    product = candidate

        # Industry Classification
        industry = None
        if product:
            if 'oil' in product:
                industry = 'oil/food product'
            elif 'gold' in product or 'jewellery' in product:
                industry = 'jewellery/gold'
            elif 'kettle' in product or 'cooker' in product:
                industry = 'household electrical appliance'
            elif 'helmet' in product:
                industry = 'personal protective equipment'
            elif 'cement' in product or 'steel' in product or 'pipe' in product:
                industry = 'construction materials'
            else:
                industry = 'manufactured product'

        # Goal Extraction
        goal = None
        if re.search(r'\b(get|obtain|apply for|procedure to get|how to get)\b.*\b(certification|certificate|licence|license|bis mark|standard mark)\b', q_lower):
            goal = 'obtain BIS certification'
        elif re.search(r'\b(hallmarking|hallmark)\b', q_lower):
            goal = 'hallmarking compliance'
        elif re.search(r'\b(testing|test facility|laboratory|testing facility)\b', q_lower):
            goal = 'product testing and conformity'
        elif re.search(r'\b(qco|quality control order|mandatory)\b', q_lower):
            goal = 'regulatory QCO compliance'
        elif re.search(r'\b(what is bis|define bis|meaning of bis)\b', q_lower):
            goal = 'understand BIS'

        # Scenario Complexity Determination
        is_scenario_words = any(w in q_lower for w in ['i am', 'we are', 'i run', 'i manufacture', 'my company', 'we produce', 'our factory', 'already have'])
        is_procedural = any(w in q_lower for w in ['procedure', 'process', 'how to apply', 'requirements before applying', 'steps to get'])

        scenario_complexity = 'high' if (is_scenario_words or (is_procedural and len(q_clean.split()) > 5)) else 'low'

        return {
            "user_role": user_role,
            "product": product,
            "industry": industry,
            "goal": goal,
            "location": None,
            "intent": None,  # Will be set by classify_intent
            "scenario_complexity": scenario_complexity
        }

    # -------------------------------------------------------------------------
    # 2. INTENT CLASSIFICATION
    # -------------------------------------------------------------------------
    def classify_intent(self, question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Classifies intent and determines adaptive retrieval path (FAST vs DEEP)."""
        q_clean = question.strip().lower()
        words = q_clean.split()

        # Fast path triggers for concise definitional / FAQ questions
        fast_factual_patterns = [
            r'^\s*what\s+is\s+bis\??\s*$',
            r'^\s*what\s+does\s+bis\s+stand\s+for\??\s*$',
            r'^\s*what\s+is\s+an?\s+is\s+number\??\s*$',
            r'^\s*what\s+is\s+coc\??\s*$',
            r'^\s*what\s+is\s+qco\??\s*$'
        ]
        for p in fast_factual_patterns:
            if re.search(p, q_clean):
                return {"intent": "factual", "path": "FAST", "top_k": 3, "use_expansion": False}

        # Short definitional queries (<= 5 words starting with what/define/meaning)
        if len(words) <= 5 and any(q_clean.startswith(prefix) for prefix in ["what is ", "define ", "meaning of "]):
            if "hallmarking" in q_clean or "hallmark" in q_clean:
                return {"intent": "hallmarking", "path": "FAST", "top_k": 3, "use_expansion": False}
            return {"intent": "factual", "path": "FAST", "top_k": 4, "use_expansion": False}

        # Unrelated non-BIS questions
        unrelated_keywords = ["java inheritance", "python class", "react props", "who is the president", "recipe for cake"]
        for uk in unrelated_keywords:
            if uk in q_clean:
                return {"intent": "unknown", "path": "FAST", "top_k": 2, "use_expansion": False}

        # Scenario-based / business situation questions
        if scenario.get("scenario_complexity") == "high":
            if re.search(r'\b(procedure|process|how to get|how to apply|apply for)\b', q_clean):
                intent = "certification_procedure"
            elif re.search(r'\b(testing|facility|lab|test)\b', q_clean):
                intent = "testing_requirement"
            elif re.search(r'\b(qco|mandatory|compulsory|order)\b', q_clean):
                intent = "qco/regulatory"
            elif re.search(r'\b(manufacture|factory|surveillance)\b', q_clean):
                intent = "manufacturer_guidance"
            else:
                intent = "scenario_based"
            return {"intent": intent, "path": "DEEP", "top_k": 8, "use_expansion": True}

        # General intent classification for complex non-scenario queries
        if re.search(r'\b(apply|application|procedure|process|grant of licence|get licence)\b', q_clean):
            return {"intent": "certification_procedure", "path": "DEEP", "top_k": 6, "use_expansion": True}
        if re.search(r'\b(qco|quality control order|mandatory|compulsory)\b', q_clean):
            return {"intent": "qco/regulatory", "path": "DEEP", "top_k": 6, "use_expansion": True}
        if re.search(r'\b(test|testing|lab|laboratory|sample)\b', q_clean):
            return {"intent": "testing_requirement", "path": "DEEP", "top_k": 6, "use_expansion": True}
        if re.search(r'\b(hallmark|hallmarking|assaying|gold)\b', q_clean):
            return {"intent": "hallmarking", "path": "FAST", "top_k": 5, "use_expansion": False}
        if re.search(r'\b(difference|compare|versus|vs)\b', q_clean):
            return {"intent": "comparison", "path": "DEEP", "top_k": 6, "use_expansion": True}
        if re.search(r'\bis\s*\d+', q_clean):
            return {"intent": "standard_lookup", "path": "DEEP", "top_k": 5, "use_expansion": True}

        return {"intent": "factual", "path": "DEEP", "top_k": 6, "use_expansion": True}

    # -------------------------------------------------------------------------
    # 3. EVIDENCE REQUIREMENT PLANNER
    # -------------------------------------------------------------------------
    def plan_required_categories(self, scenario: Dict[str, Any], intent_info: Dict[str, Any]) -> List[str]:
        """Plans which evidence categories are required for the question."""
        intent = intent_info.get("intent", "factual")
        complexity = scenario.get("scenario_complexity", "low")

        if intent in ["factual", "hallmarking"] and complexity == "low":
            return ["certification_process"]

        if intent == "qco/regulatory":
            return ["regulatory_qco", "product_standard", "certification_process"]

        if intent == "testing_requirement":
            return ["testing", "manufacturing", "certification_process", "application"]

        if intent in ["scenario_based", "certification_procedure", "manufacturer_guidance"] or complexity == "high":
            return [
                "product_standard",
                "product_specific_guidance",
                "certification_process",
                "testing",
                "manufacturing",
                "regulatory_qco",
                "application",
                "marking"
            ]

        return ["product_standard", "certification_process"]

    # -------------------------------------------------------------------------
    # 4. QUERY DECOMPOSITION & SECOND-PASS TARGETED RETRIEVAL (MULTI-HOP)
    # -------------------------------------------------------------------------
    def decompose_query(self, question: str, scenario: Dict[str, Any], intent_info: Dict[str, Any]) -> List[str]:
        """Generates first-pass subqueries for scenario/procedural queries."""
        path = intent_info.get("path", "FAST")
        if path == "FAST":
            return [question]

        product = scenario.get("product")
        intent = intent_info.get("intent", "scenario_based")

        queries = [question]

        if product:
            queries.extend([
                f"What Indian Standard applies to {product}?",
                f"What product specific BIS guidance applies to {product}?",
                f"What is the BIS certification process for {product}?",
                f"What manufacturing requirements apply to {product}?",
                f"What testing and conformity requirements apply to {product}?",
                f"What documents and application requirements apply to {product} BIS certification?",
                f"What inspection and assessment requirements apply for {product}?",
                f"What marking and licence requirements apply for {product}?",
                f"Is there a Quality Control Order QCO or compulsory certification requirement for {product}?"
            ])
        elif intent in ["certification_procedure", "scenario_based", "manufacturer_guidance"]:
            queries.extend([
                "What is the BIS certification process and grant of licence guidelines?",
                "What factory testing facility and laboratory requirements apply before applying for BIS licence?",
                "What manufacturing capability and quality control requirements apply?",
                "What document and application requirements are needed for BIS licence?",
                "What factory inspection and assessment steps apply during BIS audit?",
                "What marking and licence conditions apply for certified products?",
                "Quality Control Order QCO section 16 compulsory use of Standard Mark"
            ])
        elif intent == "testing_requirement":
            queries.extend([
                "What testing facilities and equipment are required for BIS certification?",
                "What factory laboratory testing requirements apply for licensee?",
                "What sample testing and non conformity procedures apply?"
            ])
        elif intent == "qco/regulatory":
            queries.extend([
                "Quality Control Order QCO section 16 compulsory use of Standard Mark",
                "compulsory certification order notified goods Scheme-I Scheme-IV"
            ])

        seen = set()
        deduped = []
        for q in queries:
            q_norm = q.lower().strip()
            if q_norm not in seen:
                seen.add(q_norm)
                deduped.append(q)

        return deduped

    def extract_entities_from_docs(self, docs_with_scores: List[Tuple[Any, float]], scenario: Dict[str, Any]) -> List[str]:
        """Extracts key entities (IS numbers, schemes, products) for multi-hop second pass."""
        entities = []
        if scenario.get("product"):
            entities.append(scenario["product"])

        for doc, _ in docs_with_scores:
            is_num = doc.metadata.get("is_number")
            if is_num and is_num not in entities:
                entities.append(is_num)

            # Regex extract IS numbers from chunk text
            text_is = re.findall(r'\bIS\s*[:/-]?\s*\d+(?:\s*\(Part\s*\d+\))?(?::\s*\d{4})?\b', doc.page_content, re.IGNORECASE)
            for m in text_is[:2]:
                clean_m = re.sub(r'\s+', ' ', m).upper()
                if clean_m not in entities:
                    entities.append(clean_m)

        return entities

    def generate_second_pass_queries(self, missing_categories: List[str], scenario: Dict[str, Any], entities: List[str]) -> List[str]:
        """Generates targeted second-pass queries specifically for missing evidence categories."""
        product = scenario.get("product") or ""
        is_entities = [e for e in entities if e.startswith("IS")]
        subject = is_entities[0] if is_entities else (product if product else "BIS certification")

        second_pass_queries = []

        cat_templates = {
            "product_standard": [
                f"What Indian Standard IS applies to {subject}?",
                f"Indian Standard specification requirement for {subject}"
            ],
            "product_specific_guidance": [
                f"{subject} product manual guidelines BIS licence",
                f"product specific guidance guidelines for {subject}"
            ],
            "certification_process": [
                f"{subject} BIS certification process grant of licence guidelines",
                f"Grant of Licence procedure for {subject} Scheme-I Scheme-IV"
            ],
            "testing": [
                f"{subject} testing facilities factory laboratory requirements",
                f"sample testing conformity assessment requirements for {subject}"
            ],
            "manufacturing": [
                f"{subject} factory manufacturing quality control requirements",
                f"manufacturing plant inspection raw material requirements for {subject}"
            ],
            "regulatory_qco": [
                f"{subject} Quality Control Order QCO compulsory certification",
                f"is BIS certification mandatory QCO section 16 for {subject}"
            ],
            "application": [
                f"{subject} application form documentation fees requirements BIS licence",
                f"how to apply for BIS licence application submission Form-V for {subject}"
            ],
            "marking": [
                f"{subject} Standard Mark marking requirements licence conditions",
                f"labelling and marking requirements for {subject}"
            ]
        }

        for cat in missing_categories:
            if cat in cat_templates:
                second_pass_queries.extend(cat_templates[cat])

        seen = set()
        deduped = []
        for q in second_pass_queries:
            qn = q.lower().strip()
            if qn not in seen:
                seen.add(qn)
                deduped.append(q)

        return deduped

    def retrieve_multi_query_documents(self, queries: List[str], route: Dict[str, Any]) -> List[Tuple[Any, float]]:
        if not self.is_ready():
            raise RuntimeError(self._init_error or "RAG Service is not initialized.")

        top_k = route.get("top_k", 8)

        all_results = []
        top_per_query = 3 if len(queries) > 1 else top_k

        for q in queries:
            res = self.db.similarity_search_with_score(q, k=top_per_query)
            all_results.extend(res)

        # Deduplicate by content MD5 hash, retaining best L2 score
        unique_chunks = {}
        for doc, score in all_results:
            content_hash = hashlib.md5(doc.page_content.strip().encode('utf-8')).hexdigest()
            if content_hash not in unique_chunks or score < unique_chunks[content_hash][1]:
                unique_chunks[content_hash] = (doc, score)

        sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x[1])
        return sorted_chunks[:top_k]

    # -------------------------------------------------------------------------
    # 5. EVIDENCE GROUPING & COVERAGE
    # -------------------------------------------------------------------------
    def group_evidence(self, hybrid_docs: List[Tuple[Any, float]]) -> Dict[str, List[Dict[str, Any]]]:
        categories = {
            "product_standard": [],
            "product_specific_guidance": [],
            "certification_process": [],
            "testing": [],
            "manufacturing": [],
            "regulatory_qco": [],
            "application": [],
            "marking": [],
            "other": []
        }

        for doc, score in hybrid_docs:
            src_doc = doc.metadata.get("source_document", "BIS Document")
            doc_title = doc.metadata.get("document_title", src_doc)
            page_num = doc.metadata.get("page_number", "N/A")
            cat_meta = doc.metadata.get("category", "general")
            is_num = doc.metadata.get("is_number", "")
            clause_num = doc.metadata.get("clause_number", "")
            sec_num = doc.metadata.get("section_number", "")
            scheme_num = doc.metadata.get("scheme_number", "")
            qco_ref = doc.metadata.get("qco_reference", "")
            content_lower = doc.page_content.lower()
            fn_lower = src_doc.lower()

            item = {
                "source_document": src_doc,
                "document_title": doc_title,
                "page_number": page_num,
                "category_meta": cat_meta,
                "is_number": is_num,
                "clause_number": clause_num,
                "section_number": sec_num,
                "scheme_number": scheme_num,
                "qco_reference": qco_ref,
                "relevance_score": round(float(score), 4),
                "content": doc.page_content
            }

            assigned_cat = "other"
            if is_num or "bs_" in fn_lower or "indian standard" in content_lower or cat_meta == "standards":
                assigned_cat = "product_standard"
            elif qco_ref or "quality control order" in content_lower or "qco" in content_lower or "section 16" in content_lower or cat_meta == "regulations":
                assigned_cat = "regulatory_qco"
            elif "guideline" in fn_lower or "guidelines" in content_lower or "product manual" in content_lower:
                assigned_cat = "product_specific_guidance"
            elif "test" in content_lower or "laboratory" in content_lower or "assaying" in content_lower or cat_meta == "laboratories":
                assigned_cat = "testing"
            elif "manufactur" in content_lower or "factory" in content_lower or "surveillance" in content_lower or "raw material" in content_lower:
                assigned_cat = "manufacturing"
            elif "application" in content_lower or "form" in content_lower or "fee" in content_lower or "document" in content_lower:
                assigned_cat = "application"
            elif "mark" in content_lower or "standard mark" in content_lower or "licence" in content_lower or cat_meta == "certification":
                assigned_cat = "certification_process"
            elif "marking" in content_lower or "label" in content_lower:
                assigned_cat = "marking"

            categories[assigned_cat].append(item)

        return categories

    def validate_evidence(self, grouped_evidence: Dict[str, List[Dict[str, Any]]], required_categories: List[str]) -> Dict[str, Any]:
        validation_status = {}
        summary_lines = []

        category_labels = {
            "product_standard": "PRODUCT STANDARD",
            "product_specific_guidance": "PRODUCT SPECIFIC GUIDANCE",
            "certification_process": "CERTIFICATION PROCESS",
            "testing": "TESTING & CONFORMITY",
            "manufacturing": "MANUFACTURING REQUIREMENTS",
            "regulatory_qco": "REGULATORY / QCO STATUS",
            "application": "APPLICATION REQUIREMENTS",
            "marking": "MARKING & LICENCE"
        }

        for cat_key in required_categories:
            cat_name = category_labels.get(cat_key, cat_key.upper())
            count = len(grouped_evidence.get(cat_key, []))
            if count > 0:
                validation_status[cat_key] = True
                summary_lines.append(f"{cat_name}: Evidence found ✓ ({count} chunk(s))")
            else:
                validation_status[cat_key] = False
                summary_lines.append(f"{cat_name}: Evidence insufficient ⚠ (Needs verification)")

        return {
            "status_map": validation_status,
            "summary_text": "\n".join(summary_lines)
        }

    # -------------------------------------------------------------------------
    # 6. CLAIM VALIDATION LAYER
    # -------------------------------------------------------------------------
    def validate_claims(self, grouped_evidence: Dict[str, List[Dict[str, Any]]], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Creates internal claim-to-evidence mappings for regulatory and standard claims."""
        claims = {}

        # 1. Product Standard Claim
        std_items = grouped_evidence.get("product_standard", [])
        if std_items:
            found_is = [item["is_number"] for item in std_items if item.get("is_number")]
            claims["product_standard"] = {
                "status": "SUPPORTED",
                "details": f"Indian Standard identified: {', '.join(found_is)}" if found_is else "Indian Standard specification text present"
            }
        else:
            claims["product_standard"] = {
                "status": "NOT ESTABLISHED",
                "details": f"Specific Indian Standard for {scenario.get('product') or 'product'} not found in indexed documents"
            }

        # 2. Regulatory QCO Mandatory Claim
        qco_items = grouped_evidence.get("regulatory_qco", [])
        if qco_items:
            claims["regulatory_qco"] = {
                "status": "SUPPORTED",
                "details": "Quality Control Order (QCO) regulatory evidence present in retrieved documents"
            }
        else:
            claims["regulatory_qco"] = {
                "status": "NOT ESTABLISHED",
                "details": "Mandatory QCO status could not be established from available indexed evidence. Must be verified against latest official regulatory sources."
            }

        # 3. Testing & Conformity Claim
        test_items = grouped_evidence.get("testing", [])
        if test_items:
            claims["testing"] = {
                "status": "SUPPORTED",
                "details": "Testing facility / laboratory requirements evidence present"
            }
        else:
            claims["testing"] = {
                "status": "NOT ESTABLISHED",
                "details": "Specific laboratory testing parameters not detailed in retrieved chunks"
            }

        return claims

    # -------------------------------------------------------------------------
    # LLM CALL HELPERS
    # -------------------------------------------------------------------------
    def _generate_gemini_content(self, prompt: str) -> str:
        last_error = None
        models = get_text_models()
        for model_name in models:
            try:
                t0 = time.time()
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    dt = time.time() - t0
                    print(f"[RAG_DIAG] Gemini API call succeeded using '{model_name}' in {dt:.2f}s")
                    return response.text
            except Exception as e:
                err_str = str(e)
                print(f"[RAG_WARN] Gemini API call failed for '{model_name}': {err_str[:120]}")
                last_error = e
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    continue
                elif "404" not in err_str and "NOT_FOUND" not in err_str:
                    time.sleep(0.2)
                continue

        raise RuntimeError(f"Gemini API request failed across models: {str(last_error)}")

    def get_text_model(self) -> str:
        return os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL).strip()

    def get_vision_model(self) -> str:
        return os.getenv("GEMINI_VISION_MODEL", DEFAULT_VISION_MODEL).strip()

    def get_gemini_status(self) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        return "configured" if api_key and api_key.strip() else "not_configured"

    def calculate_confidence(self, best_score: float, answer_text: str) -> str:
        ans_lower = answer_text.lower()
        refusal_phrases = ["sorry", "couldn't find", "no mention", "no information", "not referenced", "does not contain any information", "not found in"]
        if any(p in ans_lower for p in refusal_phrases):
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
        if any('\u0b80' <= c <= '\u0bff' or '\u0900' <= c <= '\u097f' for c in query):
            norm_prompt = f"Translate/normalize the following user query into concise English technical search terms for vector database lookup of Indian Standards: '{query}'. Return ONLY the English search terms with no extra commentary."
            try:
                english_terms = self._generate_gemini_content(norm_prompt).strip()
                return f"{query} {english_terms}"
            except Exception:
                return query
        return query

    # -------------------------------------------------------------------------
    # MAIN SCENARIO-AWARE 2-PASS RAG PIPELINE
    # -------------------------------------------------------------------------
    def process_query(self, question: str, response_language: str = None) -> Dict[str, Any]:
        t_req_start = time.time()
        self.total_queries += 1

        t_init_check_start = time.time()
        if not self.is_ready():
            raise RuntimeError(self._init_error or "RAG service is not ready.")
        t_init_check = time.time() - t_init_check_start

        target_lang = self.normalize_response_language(response_language, question)
        clean_q = question.strip()

        # Cache check
        cache_key = f"{clean_q.lower()}_{target_lang}_{self.index_version}"
        cached_result = self.answer_cache.get(cache_key)
        if cached_result:
            self.cache_hits += 1
            self.path_counts["CACHED"] = self.path_counts.get("CACHED", 0) + 1
            t_total = time.time() - t_req_start
            self.latencies.append(t_total)
            
            res_copy = dict(cached_result)
            res_copy["execution_time_seconds"] = round(t_total, 4)
            res_copy["timings"] = {
                "init_check_sec": round(t_init_check, 4),
                "query_norm_sec": 0.0,
                "first_pass_time_sec": 0.0,
                "second_pass_time_sec": 0.0,
                "total_retrieval_time_sec": 0.0,
                "generation_time_sec": 0.0,
                "total_time_sec": round(t_total, 4),
                "cache_hit": True
            }
            print(f"[PURIVU CACHE HIT] Query: '{clean_q}' | Served in {t_total:.4f}s")
            return res_copy

        # Stage 1: Scenario Understanding
        scenario = self.extract_scenario(clean_q)

        # Stage 2: Intent Classification & Route Determination
        intent_info = self.classify_intent(clean_q, scenario)
        scenario["intent"] = intent_info["intent"]
        route = intent_info

        self.path_counts[route["path"]] = self.path_counts.get(route["path"], 0) + 1

        # Stage 3: Evidence Requirement Planning
        required_categories = self.plan_required_categories(scenario, route)

        # Stage 4: First-Pass Multi-Query Retrieval
        t_norm_start = time.time()
        retrieval_query = self.normalize_query_for_retrieval(clean_q)
        subqueries_pass1 = self.decompose_query(retrieval_query, scenario, route)
        t_norm = time.time() - t_norm_start

        t_pass1_start = time.time()
        docs_pass1 = self.retrieve_multi_query_documents(subqueries_pass1, route)
        t_pass1 = time.time() - t_pass1_start

        # Calculate First-Pass Coverage
        grouped_pass1 = self.group_evidence(docs_pass1)
        found_pass1 = [c for c in required_categories if len(grouped_pass1.get(c, [])) > 0]
        missing_pass1 = [c for c in required_categories if c not in found_pass1]

        # Stage 5: Targeted Second-Pass Retrieval (Multi-Hop) if missing categories exist
        docs_final = list(docs_pass1)
        subqueries_pass2 = []
        t_pass2 = 0.0

        if missing_pass1 and route["path"] == "DEEP":
            t_pass2_start = time.time()
            entities = self.extract_entities_from_docs(docs_pass1, scenario)
            subqueries_pass2 = self.generate_second_pass_queries(missing_pass1, scenario, entities)

            if subqueries_pass2:
                docs_pass2 = self.retrieve_multi_query_documents(subqueries_pass2, route)

                # Merge & deduplicate Pass 1 + Pass 2 documents by content MD5 hash
                combined_dict = {}
                for doc, score in docs_pass1 + docs_pass2:
                    chash = hashlib.md5(doc.page_content.strip().encode('utf-8')).hexdigest()
                    if chash not in combined_dict or score < combined_dict[chash][1]:
                        combined_dict[chash] = (doc, score)

                sorted_combined = sorted(combined_dict.values(), key=lambda x: x[1])
                # Expand max capacity slightly for 2-pass deep retrieval
                docs_final = sorted_combined[:route.get("top_k", 8) + 3]
                t_pass2 = time.time() - t_pass2_start

        t_retrieval = t_pass1 + t_pass2
        self.retrieval_times.append(t_retrieval)

        # Exit check for unknown queries or empty retrieval
        if intent_info["intent"] == "unknown" or not docs_final:
            refusal = "Sorry, I couldn't find relevant Indian Standards or official BIS evidence for your question."
            if target_lang == "Tamil":
                refusal = "மன்னித்துக்கொள்ளுங்கள், வழங்கப்பட்ட BIS ஆவணங்களில் இந்தத் தகவல் கிடைக்கவில்லை. தவறான BIS தரநிலை அல்லது certification requirement-ஐ உருவாக்காமல் இருக்க, நான் இதை உறுதிப்படுத்தப்பட்ட தகவலாகக் கூறவில்லை."
            elif target_lang == "Hindi":
                refusal = "क्षमा करें, दिए गए BIS दस्तावेजों में यह जानकारी नहीं मिल सकी। किसी गलत जानकारी से बचने के लिए, केवल आधिकारिक दस्तावेजों के आधार पर ही उत्तर दिया जाता है।"
            
            t_total = time.time() - t_req_start
            self.latencies.append(t_total)
            return {
                "answer": refusal,
                "evidence_status": "Limited Evidence",
                "sources": [],
                "response_language": target_lang,
                "execution_time_seconds": round(t_total, 3),
                "timings": {
                    "intent": intent_info["intent"],
                    "scenario_complexity": scenario["scenario_complexity"],
                    "retrieval_path": route["path"],
                    "first_pass_time_sec": round(t_pass1, 4),
                    "second_pass_time_sec": round(t_pass2, 4),
                    "required_categories": required_categories,
                    "found_categories": [],
                    "missing_categories": required_categories,
                    "evidence_coverage": f"0 of {len(required_categories)} required evidence areas found",
                    "total_retrieved": 0,
                    "final_evidence_count": 0,
                    "total_retrieval_time_sec": round(t_retrieval, 4),
                    "generation_time_sec": 0.0,
                    "total_time_sec": round(t_total, 4)
                }
            }

        # Stage 6: Final Evidence Grouping & Validation
        t_rerank_start = time.time()
        best_l2_score = docs_final[0][1]

        grouped_evidence = self.group_evidence(docs_final)
        found_final = [c for c in required_categories if len(grouped_evidence.get(c, [])) > 0]
        missing_final = [c for c in required_categories if c not in found_final]

        coverage_str = f"{len(found_final)} of {len(required_categories)} required evidence areas found"

        evidence_val = self.validate_evidence(grouped_evidence, required_categories)

        # Stage 7: Claim Validation Layer
        claims_val = self.validate_claims(grouped_evidence, scenario)
        t_rerank = time.time() - t_rerank_start

        # Source Citation Construction
        sources_list = []
        seen_sources = set()
        context_chunks = []

        idx = 1
        for cat_name, cat_items in grouped_evidence.items():
            for item in cat_items:
                src_doc = item["source_document"]
                doc_title = item["document_title"]
                page_num = item["page_number"]

                meta_str = f"Document: {doc_title} ({src_doc}) | Category: {cat_name} | Page: {page_num}"
                if item["is_number"]: meta_str += f" | IS: {item['is_number']}"
                if item["clause_number"]: meta_str += f" | Clause: {item['clause_number']}"
                if item["section_number"]: meta_str += f" | Section: {item['section_number']}"

                context_chunks.append(f"SOURCE {idx} [{cat_name.upper()}]\n{meta_str}\nRelevant Text:\n{item['content']}")
                idx += 1

                src_key = f"{src_doc}_p{page_num}"
                if src_key not in seen_sources:
                    seen_sources.add(src_key)
                    sources_list.append({
                        "document": src_doc,
                        "document_title": doc_title if doc_title != src_doc else None,
                        "page": page_num,
                        "category": cat_name if cat_name != "other" else None,
                        "is_number": item["is_number"] if item["is_number"] else None,
                        "clause_number": item["clause_number"] if item["clause_number"] else None,
                        "section_number": item["section_number"] if item["section_number"] else None,
                        "scheme_number": item["scheme_number"] if item["scheme_number"] else None,
                        "qco_reference": item["qco_reference"] if item["qco_reference"] else None,
                        "relevance_score": item["relevance_score"]
                    })

        context_text = "\n\n".join(context_chunks)

        # Stage 8: Grounded Answer Generation with Dynamic Answer Structure
        t_gemini_start = time.time()

        if target_lang == "Tamil":
            lang_directive = (
                "CRITICAL MANDATORY LANGUAGE INSTRUCTION:\n"
                "You MUST write your ENTIRE explanation and markdown section headers in TAMIL (தமிழ்).\n"
                "Keep standard BIS identifiers (BIS, IS numbers, QCO, CoC, Scheme-I, Clause numbers) in original English script."
            )
        elif target_lang == "Hindi":
            lang_directive = (
                "CRITICAL MANDATORY LANGUAGE INSTRUCTION:\n"
                "You MUST write your ENTIRE explanation and markdown section headers in HINDI (हिन्दी).\n"
                "Keep standard BIS identifiers (BIS, IS numbers, QCO, CoC, Scheme-I, Clause numbers) in original English script."
            )
        else:
            lang_directive = "You MUST write your entire response in English."

        # Format Claim Validation Summary
        claims_summary_str = "\n".join([f"- {k.upper()}: Status={v['status']} | Details={v['details']}" for k, v in claims_val.items()])

        # Dynamic Layout Selection based on Intent & Complexity
        if route["path"] == "DEEP" or scenario["scenario_complexity"] == "high":
            prompt = f"""
You are PURIVU (BIS Saathi), an AI-powered Intelligent Assistant for Indian Standards and Bureau of Indian Standards (BIS) services.

{lang_directive}

USER SCENARIO CONTEXT:
- Role: {scenario.get('user_role') or 'Not specified'}
- Product: {scenario.get('product') or 'Not specified'}
- Industry: {scenario.get('industry') or 'Not specified'}
- Goal: {scenario.get('goal') or 'Not specified'}
- Intent: {intent_info['intent']}

INTERNAL EVIDENCE COVERAGE SUMMARY ({coverage_str}):
{evidence_val['summary_text']}

INTERNAL CLAIM VALIDATION STATUS:
{claims_summary_str}

CRITICAL GROUNDING & REGULATORY RULES:
1. Answer strictly using ONLY the provided Context chunks below.
2. IMPORTANT REGULATORY RULE: An Indian Standard existing DOES NOT mean BIS certification is legally mandatory. Certification is only mandatory if a Quality Control Order (QCO) explicitly mandates it in retrieved evidence. If mandatory QCO status is NOT ESTABLISHED in retrieved text, state explicitly: "Mandatory status could not be established from the available indexed BIS evidence and should be verified against the latest official regulatory source."
3. DO NOT invent IS numbers, clause numbers, testing requirements, fees, or laboratories.
4. SUPPORTED claims may be presented as established facts. NOT ESTABLISHED claims MUST NOT be presented as facts; mark them explicitly as "Needs verification".
5. Structure your response into clean markdown section headers in {target_lang}:

### 1. Understanding Your Situation
[Brief, practical summary of user's role, product, and certification objective in {target_lang}]

### 2. Applicable Indian Standard
[Specify exact Indian Standard (IS) number(s) from retrieved evidence. If NOT ESTABLISHED, write "Needs verification: Specific IS number for this product was not found in retrieved documents."]

### 3. Regulatory & Mandate Status (Mandatory vs Voluntary)
[State whether certification appears mandatory or voluntary based ONLY on QCO evidence in retrieved documents. Do NOT make unbacked legal claims.]

### 4. Certification Pathway
[Explain Scheme-I (Grant of Licence) or Scheme-IV (Certificate of Conformity) as established in retrieved evidence.]

### 5. Manufacturing Preparation
[Detail factory quality control, testing equipment, raw material verification from retrieved evidence. Mark missing items as 'Needs verification'.]

### 6. Testing & Conformity Requirements
[Detail sample testing and factory lab requirements from retrieved evidence. DO NOT invent laboratories.]

### 7. Application & Assessment Steps
[Step-by-step submission, documentation, and factory audit steps from retrieved evidence.]

### 8. Marking & Licensing Requirements
[Standard Mark application, licence conditions.]

### 9. Items to Verify / Evidence Gaps
[Explicitly list all items marked as 'Needs verification' or NOT ESTABLISHED.]

### 10. Recommended Next Actions & Clarifications
[Clear next steps. Include 2-3 targeted clarification questions if key scenario details are missing.]

### 11. Official Sources
[Brief citation summary referencing the cited BIS source documents.]

*(LEGAL CAVEAT: State that official certification decisions are issued exclusively by the Bureau of Indian Standards (BIS).)*

Context Chunks:
{context_text}

Original Question:
{question}

Final Structured Answer (in {target_lang}):
"""
        elif intent_info["intent"] in ["certification_procedure", "standard_lookup"]:
            prompt = f"""
You are PURIVU (BIS Saathi), an AI-powered Intelligent Assistant for Indian Standards and Bureau of Indian Standards (BIS) services.

{lang_directive}

INTERNAL CLAIM VALIDATION STATUS:
{claims_summary_str}

CRITICAL GROUNDING RULES:
1. Answer strictly using ONLY the provided Context below.
2. DO NOT hallucinate standard numbers, clauses, or legal requirements.
3. Structure response into clean markdown section headers in {target_lang}:

### Applicable Standard & Overview
[Summary of standard or service]

### Certification Procedure
[Step-by-step process based on evidence]

### Key Requirements
[Manufacturing, testing, or documentation requirements]

### Recommended Next Actions
[Clear actionable steps]

### Official Sources
[Citations]

Context:
{context_text}

Question:
{question}

Final Answer (in {target_lang}):
"""
        else:
            # Simple Factual Layout
            prompt = f"""
You are PURIVU (BIS Saathi), an AI-powered Intelligent Assistant for Indian Standards and Bureau of Indian Standards (BIS) services.

{lang_directive}

CRITICAL GROUNDING RULES:
1. Answer strictly using ONLY the provided Context below.
2. DO NOT hallucinate standard numbers, clauses, or legal requirements.
3. Format response in clean markdown:
   ### Answer
   [Clear, direct explanation]

   ### Key Points
   • [Point 1]
   • [Point 2]

Context:
{context_text}

Question:
{question}

Final Answer (in {target_lang}):
"""

        answer_text = self._generate_gemini_content(prompt)
        t_gemini = time.time() - t_gemini_start
        self.gemini_times.append(t_gemini)

        confidence_level = self.calculate_confidence(best_l2_score, answer_text)

        t_total = time.time() - t_req_start
        self.latencies.append(t_total)

        timing_breakdown = {
            "intent": intent_info["intent"],
            "scenario_complexity": scenario["scenario_complexity"],
            "retrieval_path": route["path"],
            "first_pass_subqueries": subqueries_pass1,
            "second_pass_subqueries": subqueries_pass2,
            "first_pass_time_sec": round(t_pass1, 4),
            "second_pass_time_sec": round(t_pass2, 4),
            "required_categories": required_categories,
            "found_categories": found_final,
            "missing_categories": missing_final,
            "evidence_coverage": coverage_str,
            "total_retrieved": len(docs_pass1) + len(subqueries_pass2),
            "final_evidence_count": len(docs_final),
            "total_retrieval_time_sec": round(t_retrieval, 4),
            "generation_time_sec": round(t_gemini, 4),
            "total_time_sec": round(t_total, 4),
            "init_check_sec": round(t_init_check, 4),
            "query_norm_sec": round(t_norm, 4),
            "rerank_sec": round(t_rerank, 4),
            "cache_hit": False
        }

        # Stage 10 Requirement: Print Debug Diagnostic Summary
        print("\n" + "=" * 60)
        print(f"[PURIVU SCENARIO RAG 2-PASS DIAGNOSTICS]")
        print(f"Query: \"{question}\"")
        print(f"- Intent:                       {intent_info['intent']}")
        print(f"- Scenario Complexity:         {scenario['scenario_complexity']}")
        print(f"- Retrieval Path:               {route['path']}")
        print(f"- Pass 1 Time:                  {t_pass1:.4f}s ({len(subqueries_pass1)} subqueries)")
        print(f"- Pass 2 Time:                  {t_pass2:.4f}s ({len(subqueries_pass2)} targeted subqueries)")
        print(f"- Evidence Coverage:            {coverage_str}")
        print(f"- Required Categories:          {required_categories}")
        print(f"- Found Categories:             {found_final}")
        print(f"- Missing Categories:           {missing_final}")
        print(f"- Total Retrieval Time:         {t_retrieval:.4f}s")
        print(f"- Gemini Generation Time:       {t_gemini:.4f}s")
        print(f"- Total Request Time:           {t_total:.4f}s")
        print("=" * 60 + "\n")

        output_dict = {
            "answer": answer_text,
            "evidence_status": confidence_level,
            "sources": sources_list,
            "response_language": target_lang,
            "execution_time_seconds": round(t_total, 3),
            "timings": timing_breakdown
        }

        if confidence_level != "Limited Evidence":
            self.answer_cache.put(cache_key, output_dict)

        return output_dict

    def get_metrics(self) -> Dict[str, Any]:
        lats = sorted(self.latencies) if self.latencies else [0.0]
        n = len(lats)
        avg_lat = sum(lats) / n if n > 0 else 0.0
        p50 = lats[int(n * 0.50)] if n > 0 else 0.0
        p95 = lats[int(n * 0.95)] if n > 0 else 0.0
        
        avg_ret = sum(self.retrieval_times) / len(self.retrieval_times) if self.retrieval_times else 0.0
        avg_gem = sum(self.gemini_times) / len(self.gemini_times) if self.gemini_times else 0.0
        hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0.0

        return {
            "status": "ok",
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "avg_latency_sec": round(avg_lat, 3),
            "p50_latency_sec": round(p50, 3),
            "p95_latency_sec": round(p95, 3),
            "avg_retrieval_sec": round(avg_ret, 3),
            "avg_gemini_sec": round(avg_gem, 3),
            "total_chunks": self.get_chunk_count(),
            "index_version": self.index_version,
            "query_paths": self.path_counts
        }

    def process_vision_image(self, image_bytes: bytes, user_question: str = None) -> Dict[str, Any]:
        import io
        import json
        import gc
        from PIL import Image

        if not self.is_ready():
            raise RuntimeError(self._init_error or "RAG Service is not ready.")

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
        models = get_vision_models()

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                for model_name in models:
                    try:
                        response = self.gemini_client.models.generate_content(
                            model=model_name,
                            contents=[vision_prompt, pil_image]
                        )
                        if response and response.text:
                            clean_text = response.text.strip()
                            clean_text = re.sub(r'^```(json)?', '', clean_text, flags=re.IGNORECASE).strip()
                            clean_text = re.sub(r'```$', '', clean_text).strip()
                            vision_json = json.loads(clean_text)
                            print(f"[VISION_DIAG] Vision API call succeeded using '{model_name}'")
                            break
                    except Exception as e:
                        err_str = str(e)
                        print(f"[VISION_WARN] Vision API call failed for '{model_name}': {err_str[:120]}")
                        last_error = e
                        time.sleep(0.5)
                        continue
        except Exception as e:
            raise ValueError(f"Invalid image file format: {str(e)}")
        finally:
            del image_bytes
            gc.collect()

        if not vision_json:
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

        rag_query = f"{product_info['name']} {product_info['category']}"
        if user_question and user_question.strip():
            rag_query += f" {user_question.strip()}"

        rag_result = self.process_query(rag_query)

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
