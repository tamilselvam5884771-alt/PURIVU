import sys
import os

# Unbuffer stdout
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("c:/Users/TamilHari/Desktop/BIS assistant final/faq assistant final"))

from backend.rag_service import rag_service

def is_tamil(text: str) -> bool:
    return any('\u0b80' <= c <= '\u0bff' for c in text)

def is_hindi(text: str) -> bool:
    return any('\u0900' <= c <= '\u097f' for c in text)

def run_tests():
    print("[INIT] Initializing RAG service for multilingual verification...", flush=True)
    rag_service.initialize()
    print("[INIT] RAG service initialized successfully.", flush=True)

    # Test 1: English question, Tamil response language
    q1 = "What is a Certificate of Conformity?"
    print(f"\n--- TEST 1: Question='{q1}', response_language='Tamil' ---", flush=True)
    res1 = rag_service.process_query(q1, response_language="Tamil")
    print(f"Response Language: {res1.get('response_language')}", flush=True)
    print(f"Evidence Status: {res1.get('evidence_status')}", flush=True)
    print(f"Answer Sample:\n{res1.get('answer')[:250]}...", flush=True)
    assert res1.get("response_language") == "Tamil", "Failed: response_language should be Tamil"
    assert is_tamil(res1.get("answer")), "FAILED: Output answer does not contain Tamil characters!"
    print("✓ TEST 1 PASSED: Tamil answer generated successfully.", flush=True)

    # Test 2: Hallmarking in Tamil
    q2 = "What is hallmarking?"
    print(f"\n--- TEST 2: Question='{q2}', response_language='Tamil' ---", flush=True)
    res2 = rag_service.process_query(q2, response_language="Tamil")
    print(f"Response Language: {res2.get('response_language')}", flush=True)
    print(f"Answer Sample:\n{res2.get('answer')[:250]}...", flush=True)
    assert is_tamil(res2.get("answer")), "FAILED: Output answer does not contain Tamil characters!"
    print("✓ TEST 2 PASSED: Tamil answer for hallmarking generated successfully.", flush=True)

    # Test 3: Tamil question, Tamil response language
    q3 = "இந்த பொருளுக்கு BIS சான்றிதழ் தேவையா?"
    print(f"\n--- TEST 3: Question='{q3}', response_language='Tamil' ---", flush=True)
    res3 = rag_service.process_query(q3, response_language="Tamil")
    print(f"Response Language: {res3.get('response_language')}", flush=True)
    print(f"Answer Sample:\n{res3.get('answer')[:250]}...", flush=True)
    assert is_tamil(res3.get("answer")), "FAILED: Output answer does not contain Tamil characters!"
    assert "BIS" in res3.get("answer"), "FAILED: Technical identifier 'BIS' missing from Tamil answer!"
    print("✓ TEST 3 PASSED: Tamil query produced Tamil answer with preserved technical identifiers.", flush=True)

    # Test 4: Hindi question, Hindi response language
    q4 = "यह उत्पाद BIS प्रमाणन के लिए आवश्यक है?"
    print(f"\n--- TEST 4: Question='{q4}', response_language='Hindi' ---", flush=True)
    res4 = rag_service.process_query(q4, response_language="Hindi")
    print(f"Response Language: {res4.get('response_language')}", flush=True)
    print(f"Answer Sample:\n{res4.get('answer')[:250]}...", flush=True)
    assert is_hindi(res4.get("answer")), "FAILED: Output answer does not contain Hindi/Devanagari characters!"
    print("✓ TEST 4 PASSED: Hindi query produced Hindi answer.", flush=True)

    # Test 5: English question, English response language
    q5 = "What is a BIS licence?"
    print(f"\n--- TEST 5: Question='{q5}', response_language='English' ---", flush=True)
    res5 = rag_service.process_query(q5, response_language="English")
    print(f"Response Language: {res5.get('response_language')}", flush=True)
    print(f"Answer Sample:\n{res5.get('answer')[:250]}...", flush=True)
    assert res5.get("response_language") == "English", "Failed: response_language should be English"
    print("✓ TEST 5 PASSED: English query produced English answer.", flush=True)

    # Test 6: Technical terms in Tamil (CoC, BIS)
    q6 = "What is CoC?"
    print(f"\n--- TEST 6: Question='{q6}', response_language='Tamil' ---", flush=True)
    res6 = rag_service.process_query(q6, response_language="Tamil")
    print(f"Answer Sample:\n{res6.get('answer')[:250]}...", flush=True)
    assert is_tamil(res6.get("answer")), "FAILED: Output answer does not contain Tamil characters!"
    assert "CoC" in res6.get("answer") or "BIS" in res6.get("answer"), "FAILED: Technical terms CoC/BIS not preserved!"
    print("✓ TEST 6 PASSED: Technical terms preserved in Tamil answer.", flush=True)

    # Test 7: Insufficient evidence in Tamil
    q7 = "What is the BIS standard for a fictional teleportation device?"
    print(f"\n--- TEST 7: Insufficient evidence question in Tamil ---", flush=True)
    res7 = rag_service.process_query(q7, response_language="Tamil")
    print(f"Evidence Status: {res7.get('evidence_status')}", flush=True)
    print(f"Answer Sample:\n{res7.get('answer')}", flush=True)
    assert is_tamil(res7.get("answer")), "FAILED: Refusal output is not in Tamil!"
    print("✓ TEST 7 PASSED: Insufficient evidence returned Tamil refusal.", flush=True)

    # Test 8: Backward compatibility (no response_language)
    q8 = "What is hallmarking?"
    print(f"\n--- TEST 8: Backward compatibility (response_language=None) ---", flush=True)
    res8 = rag_service.process_query(q8, response_language=None)
    print(f"Response Language: {res8.get('response_language')}", flush=True)
    assert res8.get("response_language") == "English", "Failed: Default should be English"
    print("✓ TEST 8 PASSED: Backward compatibility intact.", flush=True)

    print("\n=========================================", flush=True)
    print("ALL 8 MULTILINGUAL RAG TESTS PASSED! 🎉", flush=True)
    print("=========================================", flush=True)

if __name__ == "__main__":
    run_tests()
