"""eval/retrieval_metrics.py — Recall@k + MRR + 拒答阈值扫描"""
import os, json, statistics
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
import numpy as np, torch, chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

EVAL_PATH = "eval/data/qa_eval.jsonl"
OUT_PATH  = "eval/results/retrieval_metrics.json"
RECORDS_PATH = "eval/results/records.jsonl"
KS = [1, 3, 5]
THR_RANGE = [round(0.05 + 0.025*i, 4) for i in range(19)]  # 0.05 - 0.50

print("=== retrieval_metrics start ===\n")

data = [json.loads(l) for l in open(EVAL_PATH)]
pos_n = sum(1 for d in data if d["is_answerable"]); neg_n = len(data)-pos_n
print(f"[1/5] 评估集: 正 {pos_n} 负 {neg_n}")

print("[2/5] 加载模型...")
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)
reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512, device='cuda',
    default_activation_function=torch.nn.Sigmoid())
col = chromadb.PersistentClient(path="./chroma_db").get_collection("langchain")
TOPN = min(20, col.count())
print(f"      chunks={col.count()}, topN={TOPN}")

print("[3/5] 评估中...")
records = []
for i, d in enumerate(data):
    q_emb = emb.embed_query(d["query"])
    r = col.query(query_embeddings=[q_emb], n_results=TOPN, include=["documents"])
    emb_ids, docs = r["ids"][0], r["documents"][0]
    scores = np.array(reranker.predict([(d["query"], doc) for doc in docs]))
    order = np.argsort(-scores)
    rr_ids = [emb_ids[j] for j in order]
    records.append({
        "qid": d["qid"], "query": d["query"], "is_answerable": d["is_answerable"],
        "gold_chunk_id": d["gold_chunk_id"], "rewrite_type": d["rewrite_type"],
        "emb_top": emb_ids, "rr_top": rr_ids, "rr_top_score": float(scores[order[0]]),
    })
    if (i+1) % 20 == 0: print(f"      {i+1}/{len(data)}")

def recall(rec, k, key):
    if not rec["is_answerable"]: return None
    return int(rec["gold_chunk_id"] in rec[key][:k])
def rr_score(rec, key):
    if not rec["is_answerable"]: return None
    ids = rec[key]
    return 0.0 if rec["gold_chunk_id"] not in ids else 1.0/(ids.index(rec["gold_chunk_id"])+1)
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else 0.0

print("[4/5] 计算指标...")
results = {"summary":{}, "by_type":{}, "hard_neg":{}, "threshold_sweep":[], "optimal":{}}
for name, key in [("embedding","emb_top"), ("reranker","rr_top")]:
    s = {f"recall@{k}": mean([recall(r,k,key) for r in records]) for k in KS}
    s[f"mrr@{TOPN}"] = mean([rr_score(r,key) for r in records])
    results["summary"][name] = s

for rt in ["synonym","colloquial","scenario","typo","brief"]:
    items = [r for r in records if r["rewrite_type"]==rt]
    if items: results["by_type"][rt] = mean([recall(r,3,"rr_top") for r in items])

pos_s = [r["rr_top_score"] for r in records if r["is_answerable"]]
neg_s = [r["rr_top_score"] for r in records if not r["is_answerable"]]
results["hard_neg"] = {
    "n_pos": len(pos_s), "n_neg": len(neg_s),
    "pos_score_mean": statistics.mean(pos_s), "pos_score_min": min(pos_s),
    "neg_score_mean": statistics.mean(neg_s), "neg_score_median": statistics.median(neg_s),
    "neg_score_max": max(neg_s), "neg_score_min": min(neg_s),
}

def f1(p, r): return 2*p*r/(p+r) if (p+r)>0 else 0.0
def at_thr(thr):
    tp = sum(1 for s in pos_s if s >= thr); fn = len(pos_s)-tp
    fp = sum(1 for s in neg_s if s >= thr); tn = len(neg_s)-fp
    ans_p = tp/(tp+fp) if tp+fp else 0; ans_r = tp/(tp+fn) if tp+fn else 0
    rej_p = tn/(tn+fn) if tn+fn else 0; rej_r = tn/(tn+fp) if tn+fp else 0
    return {"thr":thr,"ans_p":ans_p,"ans_r":ans_r,"ans_f1":f1(ans_p,ans_r),
            "rej_p":rej_p,"rej_r":rej_r,"rej_f1":f1(rej_p,rej_r)}
sweep = [at_thr(t) for t in THR_RANGE]
results["threshold_sweep"] = sweep
results["optimal"] = {
    "by_ans_f1": max(sweep, key=lambda x: x["ans_f1"]),
    "by_rej_f1": max(sweep, key=lambda x: x["rej_f1"]),
}

print("[5/5] 落盘 + 打印...")
Path("eval/results").mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f: json.dump(results, f, indent=2, ensure_ascii=False)
with open(RECORDS_PATH, "w") as f:
    for r in records: f.write(json.dumps(r, ensure_ascii=False)+"\n")

print(f"\n=== Retrieval (n_pos={pos_n}, n_neg={neg_n}) ===")
print(f"{'':14s}{'Embedding':>12s}{'+Reranker':>12s}")
for k in KS:
    e = results["summary"]["embedding"][f"recall@{k}"]
    r = results["summary"]["reranker"][f"recall@{k}"]
    print(f"{'Recall@'+str(k):14s}{e*100:>11.1f}%{r*100:>11.1f}%")
e = results["summary"]["embedding"][f"mrr@{TOPN}"]
r = results["summary"]["reranker"][f"mrr@{TOPN}"]
print(f"{'MRR@'+str(TOPN):14s}{e:>12.3f}{r:>12.3f}")

print(f"\n=== Threshold sweep ===")
print(f"{'thr':>6}{'ans-P':>7}{'ans-R':>7}{'ans-F1':>8} | {'rej-P':>7}{'rej-R':>7}{'rej-F1':>8}")
for x in sweep:
    print(f"{x['thr']:>6.3f}{x['ans_p']:>7.3f}{x['ans_r']:>7.3f}{x['ans_f1']:>8.3f} | {x['rej_p']:>7.3f}{x['rej_r']:>7.3f}{x['rej_f1']:>8.3f}")

ba, br = results["optimal"]["by_ans_f1"], results["optimal"]["by_rej_f1"]
print(f"\n最优 thr (ans F1): thr={ba['thr']:.3f}  F1={ba['ans_f1']:.3f}  P={ba['ans_p']:.3f} R={ba['ans_r']:.3f}")
print(f"最优 thr (rej F1): thr={br['thr']:.3f}  F1={br['rej_f1']:.3f}  P={br['rej_p']:.3f} R={br['rej_r']:.3f}")
print(f"\n✅ {OUT_PATH}\n✅ {RECORDS_PATH}\n=== retrieval_metrics done ===")
