"""eval/llm_judge.py — Qwen2.5-7B 自评 (correctness/faithfulness/relevance, 1-5)"""
import os, json, statistics
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
from openai import OpenAI

GEN_PATH = "eval/results/generations.jsonl"
OUT_PATH = "eval/results/judge.jsonl"
SUMMARY  = "eval/results/judge_summary.json"
MODEL = "qwen2.5-7b"

JUDGE_PROMPT = """你是 RAG 答案评估专家,严格打分。

【用户问题】{query}
【检索到的上下文】
{ctx}
【标准答案】{gold}
【系统生成的答案】{ans}

按 3 个维度 1-5 分严格打分(5 分罕见,3 分=基本可接受):

correctness(对照标准答案的事实正确):
  5=完全准确  4=基本准确,有小遗漏  3=方向对但有错  2=部分错  1=错或答非所问

faithfulness(基于上下文,无幻觉):
  5=完全基于上下文  4=主要基于+合理推断  3=部分超纲  2=多处幻觉  1=完全编造

relevance(是否真回答了问题):
  5=直接回答  4=核心+次要  3=部分回答  2=偏题  1=不相关

严格 JSON 输出,无任何额外文字:
{{"correctness":X,"faithfulness":X,"relevance":X,"reason":"一句话"}}"""

def extract_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0: return None
    try: return json.loads(text[s:e+1])
    except: return None

print("=== llm_judge start ===")
gens = [json.loads(l) for l in open(GEN_PATH)]
to_judge = [g for g in gens if g["is_answerable"] and not g["refused"]]
print(f"[1/3] total={len(gens)}, judging(可答&未拒答)={len(to_judge)}")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
print(f"[2/3] judging...")
judged = []
for i, g in enumerate(to_judge):
    ctx = "\n".join(f"【{j+1}】{d}" for j, d in enumerate(g["retrieved_top3_docs"]))
    prompt = JUDGE_PROMPT.format(query=g["query"], ctx=ctx,
                                  gold=g["gold_answer"], ans=g["answer"])
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role":"user","content":prompt}],
        temperature=0.0, top_p=1.0, max_tokens=250,
    )
    out = resp.choices[0].message.content
    d = extract_json(out)
    if not d or not all(k in d for k in ["correctness","faithfulness","relevance"]):
        print(f"  ⚠ {g['qid']} parse fail: {out[:80]}")
        continue
    judged.append({
        "qid": g["qid"], "query": g["query"], "rewrite_type": g["rewrite_type"],
        "answer": g["answer"], "gold_answer": g["gold_answer"],
        "correctness": int(d["correctness"]),
        "faithfulness": int(d["faithfulness"]),
        "relevance": int(d["relevance"]),
        "reason": d.get("reason",""),
    })
    if (i+1) % 10 == 0: print(f"      {i+1}/{len(to_judge)}")

Path("eval/results").mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in judged: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def mean(xs): return round(sum(xs)/len(xs), 3) if xs else 0
def med(xs):  return statistics.median(xs) if xs else 0

summary = {"n_total": len(gens), "n_judged": len(judged)}
for dim in ["correctness","faithfulness","relevance"]:
    xs = [r[dim] for r in judged]
    summary[dim] = {"mean": mean(xs), "median": med(xs),
                    "dist": {s: xs.count(s) for s in [1,2,3,4,5]}}
summary["by_type"] = {}
for rt in ["synonym","colloquial","scenario","typo","brief"]:
    items = [r for r in judged if r["rewrite_type"]==rt]
    if items:
        summary["by_type"][rt] = {
            "n": len(items),
            "correctness": mean([r["correctness"] for r in items]),
            "faithfulness": mean([r["faithfulness"] for r in items]),
            "relevance": mean([r["relevance"] for r in items]),
        }

pos = [g for g in gens if g["is_answerable"]]
neg = [g for g in gens if not g["is_answerable"]]
pos_ref = sum(1 for g in pos if g["refused"])
neg_ref = sum(1 for g in neg if g["refused"])
summary["reject_behavior"] = {
    "thr": 0.15,
    "false_reject_rate": round(pos_ref/len(pos),3),
    "correct_reject_rate": round(neg_ref/len(neg),3),
    "miss_reject_rate": round(1-neg_ref/len(neg),3),
    "rej_precision": round(neg_ref/(neg_ref+pos_ref) if (neg_ref+pos_ref) else 0, 3),
    "rej_recall": round(neg_ref/len(neg),3),
}
with open(SUMMARY, "w") as f: json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n[3/3] ✅ {OUT_PATH}, {SUMMARY}")
print(f"\n=== LLM-as-judge (Qwen 自评, n={len(judged)}) ===")
for dim in ["correctness","faithfulness","relevance"]:
    s = summary[dim]
    print(f"  {dim:14s}  mean={s['mean']:.2f}  median={s['median']}  dist={s['dist']}")

print(f"\n=== By rewrite_type ===")
print(f"  {'type':<12}{'n':>4}{'corr':>7}{'faith':>7}{'rele':>7}")
for rt, v in summary["by_type"].items():
    print(f"  {rt:<12}{v['n']:>4}{v['correctness']:>7.2f}{v['faithfulness']:>7.2f}{v['relevance']:>7.2f}")

print(f"\n=== 拒答行为 (thr=0.15) ===")
rb = summary["reject_behavior"]
print(f"  误拒={rb['false_reject_rate']*100:.1f}%  正拒={rb['correct_reject_rate']*100:.1f}%  漏拒={rb['miss_reject_rate']*100:.1f}%")
print(f"  拒答 P={rb['rej_precision']:.3f}  R={rb['rej_recall']:.3f}")

print(f"\n=== 评分最低的 3 条 ===")
worst = sorted(judged, key=lambda r: r["correctness"]+r["faithfulness"]+r["relevance"])[:3]
for r in worst:
    print(f"  [{r['rewrite_type']}] {r['query']}")
    print(f"    gen : {r['answer'][:70]}")
    print(f"    gold: {r['gold_answer'][:70]}")
    print(f"    c={r['correctness']} f={r['faithfulness']} r={r['relevance']}  ⤷ {r['reason']}")

print("=== llm_judge done ===")
