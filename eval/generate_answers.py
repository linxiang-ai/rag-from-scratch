"""eval/generate_answers.py — 跑完整 RAG pipeline (阈值=0.15 拒答 + Qwen 生成)"""
import os, json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
import chromadb
from openai import OpenAI

RECORDS_PATH = "eval/results/records.jsonl"
EVAL_PATH    = "eval/data/qa_eval.jsonl"
OUT_PATH     = "eval/results/generations.jsonl"
REJECT_THR = 0.15
TOP_K = 3
MODEL = "qwen2.5-7b"

SYSTEM_PROMPT = """你是 Shopee 电商平台的客服助手。请根据下面提供的 FAQ 上下文回答用户问题：
- 如果上下文里有明确答案，简洁准确地回复（不超过 80 字）
- 如果上下文不足以回答，回复"抱歉，我没有相关信息，建议联系人工客服"
- 不要编造上下文之外的信息或政策"""
REFUSE_MSG = "抱歉，我没有相关信息，建议联系人工客服"

print("=== generate_answers start ===")

records = [json.loads(l) for l in open(RECORDS_PATH)]
qa = {d["qid"]: d for d in (json.loads(l) for l in open(EVAL_PATH))}
print(f"[1/3] records={len(records)}, eval={len(qa)}")

col = chromadb.PersistentClient(path="./chroma_db").get_collection("langchain")
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

print(f"[2/3] 生成中 (thr={REJECT_THR}, top_k={TOP_K})...")
results = []
for i, r in enumerate(records):
    top_ids = r["rr_top"][:TOP_K]
    got = col.get(ids=top_ids, include=["documents"])
    id2doc = dict(zip(got["ids"], got["documents"]))
    docs = [id2doc[x] for x in top_ids]  # 按 reranker 顺序

    if r["rr_top_score"] < REJECT_THR:
        ans, refused = REFUSE_MSG, True
    else:
        ctx = "\n\n".join(f"【{j+1}】{d}" for j, d in enumerate(docs))
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":f"上下文:\n{ctx}\n\n用户问题: {r['query']}"}],
            temperature=0.1, top_p=0.9, max_tokens=200,
        )
        ans = resp.choices[0].message.content.strip()
        refused = ans.startswith("抱歉") or "联系人工客服" in ans

    results.append({
        "qid": r["qid"], "query": r["query"],
        "is_answerable": r["is_answerable"], "rewrite_type": r["rewrite_type"],
        "gold_chunk_id": r["gold_chunk_id"],
        "gold_question": qa[r["qid"]].get("gold_question"),
        "gold_answer": qa[r["qid"]].get("gold_answer"),
        "rr_top_score": r["rr_top_score"],
        "retrieved_top3_ids": top_ids,
        "retrieved_top3_docs": docs,
        "refused": refused, "answer": ans,
    })
    if (i+1) % 10 == 0: print(f"      {i+1}/{len(records)}")

Path("eval/results").mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

pos = [r for r in results if r["is_answerable"]]
neg = [r for r in results if not r["is_answerable"]]
pos_ref = sum(1 for r in pos if r["refused"])
neg_ref = sum(1 for r in neg if r["refused"])
print(f"\n[3/3] ✅ {OUT_PATH}: {len(results)} 条")
print(f"\n=== 拒答行为 (thr={REJECT_THR}) ===")
print(f"  误拒(正被拒): {pos_ref}/{len(pos)} = {pos_ref/len(pos)*100:.1f}%")
print(f"  正拒(负被拒): {neg_ref}/{len(neg)} = {neg_ref/len(neg)*100:.1f}%")
print(f"  漏拒(负未拒): {len(neg)-neg_ref}/{len(neg)} = {(len(neg)-neg_ref)/len(neg)*100:.1f}%")

print(f"\n=== 样例 (正样本) ===")
for r in results[:3]:
    print(f"  [{r['rewrite_type']:11s}] Q: {r['query']}")
    print(f"    gold: {r['gold_answer'][:60]}...")
    print(f"    gen : {r['answer'][:80]}")
print(f"\n=== 样例 (hard neg) ===")
for r in [x for x in results if not x['is_answerable']][:3]:
    print(f"  Q: {r['query']}  (refused={r['refused']}, score={r['rr_top_score']:.3f})")
    print(f"    gen: {r['answer'][:80]}")
print("=== generate_answers done ===")
