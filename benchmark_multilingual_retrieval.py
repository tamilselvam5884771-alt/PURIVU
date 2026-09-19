import sys
import os
import time
import json
from typing import List, Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("c:/Users/TamilHari/Desktop/BIS assistant final/faq assistant final"))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.rag_service import E5Embeddings

# Define 30 Test Queries (10 English, 10 Tamil, 10 Hindi)
BENCHMARK_QUERIES = [
    # ENGLISH (1-10)
    {"id": 1, "lang": "English", "query": "What is BIS?", "expected": ["bis", "bureau of indian standards", "act"]},
    {"id": 2, "lang": "English", "query": "What is hallmarking?", "expected": ["hallmarking", "jewellery", "gold", "assaying"]},
    {"id": 3, "lang": "English", "query": "How do I apply for BIS certification?", "expected": ["grant", "licence", "application", "form"]},
    {"id": 4, "lang": "English", "query": "What is the procedure to get BIS certification for coconut oil?", "expected": ["licence", "coc", "conformity", "testing", "oil"]},
    {"id": 5, "lang": "English", "query": "What BIS requirements should a manufacturer prepare before applying?", "expected": ["quality", "testing", "laboratory", "plant"]},
    {"id": 6, "lang": "English", "query": "Which products require mandatory BIS certification under Quality Control Orders?", "expected": ["quality control order", "qco", "section 16", "mandatory"]},
    {"id": 7, "lang": "English", "query": "What are the guidelines for Assaying and Hallmarking Centres?", "expected": ["assaying", "hallmarking", "center", "ahc"]},
    {"id": 8, "lang": "English", "query": "What is the procedure for renewal of BIS licence?", "expected": ["renewal", "licence", "validity", "form"]},
    {"id": 9, "lang": "English", "query": "How does BIS handle non-conformity and sample failure?", "expected": ["non-conformity", "failure", "sample", "stop marking"]},
    {"id": 10, "lang": "English", "query": "What is Java inheritance?", "expected": ["java", "inheritance"]}, # Unrelated

    # TAMIL (11-20)
    {"id": 11, "lang": "Tamil", "query": "BIS என்றால் என்ன?", "expected": ["bis", "bureau of indian standards", "act"]},
    {"id": 12, "lang": "Tamil", "query": "ஹால்மார்க்கிங் என்றால் என்ன?", "expected": ["hallmarking", "jewellery", "gold", "assaying"]},
    {"id": 13, "lang": "Tamil", "query": "BIS சான்றிதழுக்கு எப்படி விண்ணப்பிப்பது?", "expected": ["grant", "licence", "application", "form"]},
    {"id": 14, "lang": "Tamil", "query": "தேங்காய் எண்ணெய்க்கு BIS சான்றிதழ் பெறுவதற்கான நடைமுறை என்ன?", "expected": ["licence", "coc", "conformity", "testing", "oil"]},
    {"id": 15, "lang": "Tamil", "query": "உற்பத்தியாளர் விண்ணப்பிப்பதற்கு முன் என்னென்ன தேவைகளை ஆயத்தப்படுத்த வேண்டும்?", "expected": ["quality", "testing", "laboratory", "plant"]},
    {"id": 16, "lang": "Tamil", "query": "கட்டாய BIS சான்றிதழ் வழங்கும் Quality Control Order பற்றிய விவரங்கள் என்ன?", "expected": ["quality control order", "qco", "section 16", "mandatory"]},
    {"id": 17, "lang": "Tamil", "query": "அசேயிங் மற்றும் ஹால்மார்க்கிங் மையங்களுக்கான வழிகாட்டுதல்கள் என்ன?", "expected": ["assaying", "hallmarking", "ahc", "centre"]},
    {"id": 18, "lang": "Tamil", "query": "BIS உரிமம் புதுப்பித்தலுக்கான நடைமுறை என்ன?", "expected": ["renewal", "licence", "validity"]},
    {"id": 19, "lang": "Tamil", "query": "மாதிரி தோல்வியடைந்தால் BIS என்ன நடவடிக்கை எடுக்கும்?", "expected": ["non-conformity", "failure", "sample"]},
    {"id": 20, "lang": "Tamil", "query": "Java inheritance என்றால் என்ன?", "expected": ["java", "inheritance"]}, # Unrelated

    # HINDI (21-30)
    {"id": 21, "lang": "Hindi", "query": "BIS क्या है?", "expected": ["bis", "bureau of indian standards", "act"]},
    {"id": 22, "lang": "Hindi", "query": "हॉलमार्किंग क्या है?", "expected": ["hallmarking", "jewellery", "gold", "assaying"]},
    {"id": 23, "lang": "Hindi", "query": "BIS प्रमाणन के लिए आवेदन कैसे करें?", "expected": ["grant", "licence", "application", "form"]},
    {"id": 24, "lang": "Hindi", "query": "नारियल तेल के लिए BIS प्रमाणन प्राप्त करने की प्रक्रिया क्या है?", "expected": ["licence", "coc", "conformity", "testing", "oil"]},
    {"id": 25, "lang": "Hindi", "query": "आवेदन करने से पहले निर्माता को किन आवश्यकताओं की तैयारी करनी चाहिए?", "expected": ["quality", "testing", "laboratory", "plant"]},
    {"id": 26, "lang": "Hindi", "query": "गुणवत्ता नियंत्रण आदेश (QCO) के तहत किन उत्पादों को अनिवार्य BIS प्रमाणन की आवश्यकता है?", "expected": ["quality control order", "qco", "section 16", "mandatory"]},
    {"id": 27, "lang": "Hindi", "query": "परख और हॉलमार्किंग केंद्रों (AHC) के लिए क्या दिशानिर्देश हैं?", "expected": ["assaying", "hallmarking", "ahc", "centre"]},
    {"id": 28, "lang": "Hindi", "query": "BIS लाइसेंस के नवीनीकरण की प्रक्रिया क्या है?", "expected": ["renewal", "licence", "validity"]},
    {"id": 29, "lang": "Hindi", "query": "यदि नमूना विफल हो जाता है तो BIS क्या कार्रवाई करता है?", "expected": ["non-conformity", "failure", "sample"]},
    {"id": 30, "lang": "Hindi", "query": "जावा इनहेरिटेंस क्या है?", "expected": ["java", "inheritance"]} # Unrelated
]

def evaluate_retrieval(db: FAISS, model_name: str, is_e5: bool) -> Dict[str, Any]:
    print(f"\nEvaluating Model: '{model_name}' (E5 format: {is_e5})...", flush=True)
    
    results_by_lang = {"English": [], "Tamil": [], "Hindi": []}
    emb_latencies = []
    ret_latencies = []

    for item in BENCHMARK_QUERIES:
        q_text = item["query"]
        lang = item["lang"]
        expected_terms = item["expected"]
        is_unrelated = (item["id"] in [10, 20, 30])

        t0 = time.time()
        res_docs = db.similarity_search_with_score(q_text, k=5)
        t1 = time.time() - t0
        ret_latencies.append(t1)

        retrieved_texts = [doc.page_content.lower() for doc, _ in res_docs]
        scores = [float(score) for _, score in res_docs]

        # Check Recall@3 and Recall@5
        hit_at_3 = False
        hit_at_5 = False

        if is_unrelated:
            # For unrelated query, hit = correctly finding no relevant BIS term
            has_bis_term = any(any(exp in text for exp in ["bis", "hallmark", "licence"]) for text in retrieved_texts[:3])
            hit_at_3 = not has_bis_term
            hit_at_5 = not has_bis_term
        else:
            for idx, text in enumerate(retrieved_texts):
                matched = any(term in text for term in expected_terms)
                if matched:
                    if idx < 3:
                        hit_at_3 = True
                    if idx < 5:
                        hit_at_5 = True

        results_by_lang[lang].append({
            "id": item["id"],
            "query": q_text,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "best_score": round(scores[0], 4) if scores else None,
            "latency_sec": round(t1, 4)
        })

    metrics = {}
    for lang in ["English", "Tamil", "Hindi"]:
        lang_res = results_by_lang[lang]
        r3 = sum(1 for x in lang_res if x["hit_at_3"]) / len(lang_res) * 100.0
        r5 = sum(1 for x in lang_res if x["hit_at_5"]) / len(lang_res) * 100.0
        avg_lat = sum(x["latency_sec"] for x in lang_res) / len(lang_res)
        metrics[lang] = {
            "recall_at_3_pct": round(r3, 1),
            "recall_at_5_pct": round(r5, 1),
            "avg_latency_sec": round(avg_lat, 4)
        }

    overall_r3 = sum(1 for l in results_by_lang.values() for x in l if x["hit_at_3"]) / len(BENCHMARK_QUERIES) * 100.0
    overall_r5 = sum(1 for l in results_by_lang.values() for x in l if x["hit_at_5"]) / len(BENCHMARK_QUERIES) * 100.0
    overall_lat = sum(ret_latencies) / len(ret_latencies)

    metrics["overall"] = {
        "recall_at_3_pct": round(overall_r3, 1),
        "recall_at_5_pct": round(overall_r5, 1),
        "avg_retrieval_sec": round(overall_lat, 4)
    }

    return metrics, results_by_lang

def run_benchmark():
    print("\n=========================================================", flush=True)
    print("📊 PURIVU CROSS-LINGUAL RETRIEVAL BENCHMARK (OLD vs NEW)", flush=True)
    print("=========================================================\n", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    new_dir = os.path.join(base_dir, "faiss_index_bis")
    old_dir = os.path.join(base_dir, "faiss_index_bis_minilm_backup")

    # 1. Evaluate NEW Model (intfloat/multilingual-e5-base)
    print(f"[NEW MODEL] Loading FAISS index from '{new_dir}'...", flush=True)
    e5_embeddings = E5Embeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    new_db = FAISS.load_local(new_dir, e5_embeddings, allow_dangerous_deserialization=True)
    new_metrics, new_details = evaluate_retrieval(new_db, "intfloat/multilingual-e5-base (768d)", is_e5=True)

    # 2. Evaluate OLD Model (sentence-transformers/all-MiniLM-L6-v2) if backup exists
    old_metrics = None
    if os.path.exists(old_dir):
        print(f"\n[OLD MODEL] Loading FAISS index from '{old_dir}'...", flush=True)
        old_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        old_db = FAISS.load_local(old_dir, old_embeddings, allow_dangerous_deserialization=True)
        old_metrics, old_details = evaluate_retrieval(old_db, "sentence-transformers/all-MiniLM-L6-v2 (384d)", is_e5=False)

    print("\n" + "=" * 65, flush=True)
    print("🏆 BENCHMARK RESULTS SUMMARY (Recall@K & Latency Comparison)", flush=True)
    print("=" * 65, flush=True)

    print(f"\n--- NEW MODEL: intfloat/multilingual-e5-base (768d) ---", flush=True)
    print(f"  • English: Recall@3 = {new_metrics['English']['recall_at_3_pct']}%, Recall@5 = {new_metrics['English']['recall_at_5_pct']}%", flush=True)
    print(f"  • Tamil:   Recall@3 = {new_metrics['Tamil']['recall_at_3_pct']}%, Recall@5 = {new_metrics['Tamil']['recall_at_5_pct']}%", flush=True)
    print(f"  • Hindi:   Recall@3 = {new_metrics['Hindi']['recall_at_3_pct']}%, Recall@5 = {new_metrics['Hindi']['recall_at_5_pct']}%", flush=True)
    print(f"  • Overall: Recall@3 = {new_metrics['overall']['recall_at_3_pct']}%, Recall@5 = {new_metrics['overall']['recall_at_5_pct']}%", flush=True)
    print(f"  • Avg Retrieval Latency: {new_metrics['overall']['avg_retrieval_sec']:.4f}s", flush=True)

    if old_metrics:
        print(f"\n--- OLD MODEL: sentence-transformers/all-MiniLM-L6-v2 (384d) ---", flush=True)
        print(f"  • English: Recall@3 = {old_metrics['English']['recall_at_3_pct']}%, Recall@5 = {old_metrics['English']['recall_at_5_pct']}%", flush=True)
        print(f"  • Tamil:   Recall@3 = {old_metrics['Tamil']['recall_at_3_pct']}%, Recall@5 = {old_metrics['Tamil']['recall_at_5_pct']}%", flush=True)
        print(f"  • Hindi:   Recall@3 = {old_metrics['Hindi']['recall_at_3_pct']}%, Recall@5 = {old_metrics['Hindi']['recall_at_5_pct']}%", flush=True)
        print(f"  • Overall: Recall@3 = {old_metrics['overall']['recall_at_3_pct']}%, Recall@5 = {old_metrics['overall']['recall_at_5_pct']}%", flush=True)
        print(f"  • Avg Retrieval Latency: {old_metrics['overall']['avg_retrieval_sec']:.4f}s", flush=True)

        print("\n" + "=" * 65, flush=True)
        print("📈 RETRIEVAL IMPROVEMENT SUMMARY:", flush=True)
        print(f"  • English Recall@3 Gain: {new_metrics['English']['recall_at_3_pct'] - old_metrics['English']['recall_at_3_pct']:+.1f}%", flush=True)
        print(f"  • Tamil Recall@3 Gain:   {new_metrics['Tamil']['recall_at_3_pct'] - old_metrics['Tamil']['recall_at_3_pct']:+.1f}%", flush=True)
        print(f"  • Hindi Recall@3 Gain:   {new_metrics['Hindi']['recall_at_3_pct'] - old_metrics['Hindi']['recall_at_3_pct']:+.1f}%", flush=True)
        print(f"  • Overall Recall@3 Gain: {new_metrics['overall']['recall_at_3_pct'] - old_metrics['overall']['recall_at_3_pct']:+.1f}%", flush=True)
        print("=" * 65 + "\n", flush=True)

    # Save benchmark results to file
    out_file = os.path.join(base_dir, "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"new_model": new_metrics, "old_model": old_metrics}, f, indent=2)
    print(f"Benchmark output saved to '{out_file}'.\n", flush=True)

if __name__ == "__main__":
    run_benchmark()
