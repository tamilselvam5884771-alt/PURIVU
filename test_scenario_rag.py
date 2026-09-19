import sys
import os
import time

# Ensure UTF-8 output encoding for console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend package import path
sys.path.insert(0, os.path.abspath("c:/Users/TamilHari/Desktop/BIS assistant final/faq assistant final"))

from backend.rag_service import rag_service

def run_tests():
    print("\n=======================================================", flush=True)
    print("🚀 PURIVU SCENARIO-AWARE 2-PASS RAG SYSTEM TEST SUITE", flush=True)
    print("=======================================================\n", flush=True)

    print("[INIT] Warm-starting RAG service...", flush=True)
    rag_service.initialize()
    print("[INIT] RAG service initialized successfully.\n", flush=True)

    # -------------------------------------------------------------------------
    # TEST 1: Simple factual definition ("What is BIS?")
    # -------------------------------------------------------------------------
    q1 = "What is BIS?"
    print(f"--- TEST 1: Question = '{q1}' ---", flush=True)
    t0 = time.time()
    res1 = rag_service.process_query(q1)
    t1 = time.time() - t0
    timings1 = res1.get("timings", {})
    
    print(f"  • Retrieval Path: {timings1.get('retrieval_path')}")
    print(f"  • Intent: {timings1.get('intent')}")
    print(f"  • Evidence Coverage: {timings1.get('evidence_coverage')}")
    print(f"  • Latency: {t1:.2f}s")
    print(f"  • Answer Sample:\n{res1.get('answer')[:180]}...\n")
    
    assert timings1.get("retrieval_path") == "FAST", f"Expected FAST path, got {timings1.get('retrieval_path')}"
    print("✓ TEST 1 PASSED: Simple factual query correctly took FAST path.\n", flush=True)

    # -------------------------------------------------------------------------
    # TEST 2: Domain term lookup ("What is hallmarking?")
    # -------------------------------------------------------------------------
    q2 = "What is hallmarking?"
    print(f"--- TEST 2: Question = '{q2}' ---", flush=True)
    t0 = time.time()
    res2 = rag_service.process_query(q2)
    t2 = time.time() - t0
    timings2 = res2.get("timings", {})
    
    print(f"  • Retrieval Path: {timings2.get('retrieval_path')}")
    print(f"  • Intent: {timings2.get('intent')}")
    print(f"  • Evidence Status: {res2.get('evidence_status')}")
    print(f"  • Latency: {t2:.2f}s")
    print(f"  • Sources Cited: {len(res2.get('sources', []))}")
    print(f"  • Answer Sample:\n{res2.get('answer')[:180]}...\n")
    
    assert timings2.get("retrieval_path") == "FAST", f"Expected FAST path, got {timings2.get('retrieval_path')}"
    print("✓ TEST 2 PASSED: Domain lookup correctly took FAST path.\n", flush=True)

    # -------------------------------------------------------------------------
    # TEST 3: General procedural query ("How do I apply for BIS certification?")
    # -------------------------------------------------------------------------
    q3 = "How do I apply for BIS certification?"
    print(f"--- TEST 3: Question = '{q3}' ---", flush=True)
    t0 = time.time()
    res3 = rag_service.process_query(q3)
    t3 = time.time() - t0
    timings3 = res3.get("timings", {})
    
    print(f"  • Retrieval Path: {timings3.get('retrieval_path')}")
    print(f"  • Intent: {timings3.get('intent')}")
    print(f"  • First-Pass Subqueries: {len(timings3.get('first_pass_subqueries', []))}")
    print(f"  • Second-Pass Subqueries: {len(timings3.get('second_pass_subqueries', []))}")
    print(f"  • Evidence Coverage: {timings3.get('evidence_coverage')}")
    print(f"  • Latency: {t3:.2f}s")
    print(f"  • Answer Sample:\n{res3.get('answer')[:220]}...\n")
    
    assert timings3.get("retrieval_path") == "DEEP", f"Expected DEEP path, got {timings3.get('retrieval_path')}"
    print("✓ TEST 3 PASSED: Procedural query took DEEP path with multi-query decomposition.\n", flush=True)

    # -------------------------------------------------------------------------
    # TEST 4: Coconut Oil Business Scenario (Deep 2-Pass Verification)
    # -------------------------------------------------------------------------
    q4 = "I am an oil producer. I run a coconut oil company. What is the procedure to get BIS certification?"
    print(f"--- TEST 4 (COCONUT OIL SCENARIO): Question = '{q4}' ---", flush=True)
    t0 = time.time()
    res4 = rag_service.process_query(q4)
    t4 = time.time() - t0
    timings4 = res4.get("timings", {})
    
    scenario4 = rag_service.extract_scenario(q4)
    print(f"  • Extracted Scenario: {scenario4}")
    print(f"  • Retrieval Path: {timings4.get('retrieval_path')}")
    print(f"  • Intent: {timings4.get('intent')}")
    print(f"  • First-Pass Subqueries ({len(timings4.get('first_pass_subqueries', []))}):")
    for sq in timings4.get('first_pass_subqueries', [])[:3]:
        print(f"      - {sq}")
    print(f"  • Second-Pass Targeted Subqueries ({len(timings4.get('second_pass_subqueries', []))}):")
    for sq in timings4.get('second_pass_subqueries', []):
        print(f"      - {sq}")
    print(f"  • Required Categories: {timings4.get('required_categories')}")
    print(f"  • Found Categories:    {timings4.get('found_categories')}")
    print(f"  • Missing Categories:  {timings4.get('missing_categories')}")
    print(f"  • Evidence Coverage:   {timings4.get('evidence_coverage')}")
    print(f"  • First Pass Time: {timings4.get('first_pass_time_sec'):.4f}s | Second Pass Time: {timings4.get('second_pass_time_sec'):.4f}s")
    print(f"  • Total Latency: {t4:.2f}s")
    print(f"  • Sources Cited ({len(res4.get('sources', []))}):")
    for s in res4.get('sources', [])[:3]:
        print(f"      - {s['document']} (Page {s['page']})")
    print(f"\n  • Full Grounded Answer:\n{'-'*40}\n{res4.get('answer')}\n{'-'*40}\n")
    
    assert scenario4.get("user_role") == "producer", f"Expected producer role, got {scenario4.get('user_role')}"
    assert scenario4.get("product") == "coconut oil", f"Expected coconut oil product, got {scenario4.get('product')}"
    assert timings4.get("retrieval_path") == "DEEP", f"Expected DEEP path for scenario"
    assert len(res4.get("sources", [])) > 0, "Sources must not be empty"
    assert "mandatory status" in res4.get("answer").lower() or "verification" in res4.get("answer").lower(), "Must include mandatory verification caveat"
    print("✓ TEST 4 PASSED: Coconut oil scenario executed DEEP 2-Pass Scenario-Aware RAG successfully.\n", flush=True)

    # -------------------------------------------------------------------------
    # TEST 5: Gap-Oriented Manufacturer Scenario
    # -------------------------------------------------------------------------
    q5 = "I manufacture a product and already have a testing facility. What BIS requirements should I prepare before applying?"
    print(f"--- TEST 5 (GAP-ORIENTED SCENARIO): Question = '{q5}' ---", flush=True)
    t0 = time.time()
    res5 = rag_service.process_query(q5)
    t5 = time.time() - t0
    timings5 = res5.get("timings", {})
    scenario5 = rag_service.extract_scenario(q5)
    
    print(f"  • Extracted Scenario: {scenario5}")
    print(f"  • Retrieval Path: {timings5.get('retrieval_path')}")
    print(f"  • Intent: {timings5.get('intent')}")
    print(f"  • Evidence Coverage: {timings5.get('evidence_coverage')}")
    print(f"  • Latency: {t5:.2f}s")
    print(f"  • Answer Sample:\n{res5.get('answer')[:250]}...\n")
    
    assert timings5.get("retrieval_path") == "DEEP", f"Expected DEEP path for gap-oriented scenario"
    print("✓ TEST 5 PASSED: Gap-oriented scenario executed DEEP Scenario-Aware RAG.\n", flush=True)

    # -------------------------------------------------------------------------
    # TEST 6: Regulatory / QCO Question
    # -------------------------------------------------------------------------
    q6 = "Which products require mandatory BIS certification under Quality Control Orders?"
    print(f"--- TEST 6 (REGULATORY / QCO QUESTION): Question = '{q6}' ---", flush=True)
    t0 = time.time()
    res6 = rag_service.process_query(q6)
    t6 = time.time() - t0
    timings6 = res6.get("timings", {})
    
    print(f"  • Intent: {timings6.get('intent')}")
    print(f"  • Evidence Coverage: {timings6.get('evidence_coverage')}")
    print(f"  • Evidence Status: {res6.get('evidence_status')}")
    print(f"  • Latency: {t6:.2f}s")
    print(f"  • Answer Sample:\n{res6.get('answer')[:250]}...\n")
    
    assert timings6.get("intent") in ["qco/regulatory", "certification_procedure"], f"Expected QCO/regulatory intent"
    print("✓ TEST 6 PASSED: Regulatory / QCO question processed correctly.\n", flush=True)

    # -------------------------------------------------------------------------
    # TEST 7: Unrelated non-BIS question ("What is Java inheritance?")
    # -------------------------------------------------------------------------
    q7 = "What is Java inheritance?"
    print(f"--- TEST 7 (UNRELATED QUESTION): Question = '{q7}' ---", flush=True)
    t0 = time.time()
    res7 = rag_service.process_query(q7)
    t7 = time.time() - t0
    timings7 = res7.get("timings", {})
    
    print(f"  • Intent: {timings7.get('intent')}")
    print(f"  • Evidence Status: {res7.get('evidence_status')}")
    print(f"  • Refusal Answer: {res7.get('answer')}")
    
    assert res7.get("evidence_status") == "Limited Evidence", f"Expected Limited Evidence, got {res7.get('evidence_status')}"
    refusal_phrases = ["sorry", "couldn't find", "no mention", "no information", "not referenced", "does not contain any information", "not found in"]
    assert any(p in res7.get("answer").lower() for p in refusal_phrases), "Expected polite refusal message"
    print("✓ TEST 7 PASSED: Unrelated query produced safe refusal with Limited Evidence status.\n", flush=True)

    print("=======================================================", flush=True)
    print("🎉 ALL 7 SCENARIO-AWARE 2-PASS RAG TESTS PASSED!", flush=True)
    print("=======================================================\n", flush=True)

if __name__ == "__main__":
    run_tests()
