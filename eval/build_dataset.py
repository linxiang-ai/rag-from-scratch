"""eval/build_dataset.py — 基于 16 条 FAQ 改写生成 ~100 条评估集"""
import os, json, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
import chromadb
from openai import OpenAI

CHROMA_PATH = "./chroma_db"
OUT = "eval/data/qa_eval.jsonl"
MODEL = "qwen2.5-7b"

REWRITE_PROMPT = """你是 RAG 评估集构造员。基于下面这条 FAQ，生成 6 种不同表达的用户问法。

【FAQ】
标准问题: {q}
标准答案: {a}

【6 种类型,每种必须生成 1 条】
- synonym(同义改写,正式)
- colloquial(口语化、随意)
- scenario(具体使用场景,如"我刚买了 xx,怎么 xx")
- typo(错别字/不规范输入,模拟手机打字)
- brief(极简,1-3 个关键词)

【约束】
- 改写后仍能用同一答案回答
- 尽量不直接抄答案里的特有术语/数字
- 6 条之间要明显不同

【输出】严格 JSON,无任何额外文字:

HARD_NEG_PROMPT = """生成 15 个电商平台用户可能问、但下面 FAQ【完全没覆盖】的问题。

【已有 FAQ】
{faqs}

【要求】
- 真实电商场景(开店/卖家咨询/营销/技术对接等)
- 不能用上面任何 FAQ 答出
- 多样化、长度 5-20 字

【输出】严格 JSON: {{"queries":["...","...",...]}}"""

def extract_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0: return None
    try: return json.loads(text[s:e+1])
    except: return None

def call_llm(client, prompt, temp=0.7):
    return client.chat.completions.create(
        model=MODEL, messages=[{"role":"user","content":prompt}],
        temperature=temp, top_p=0.9, max_tokens=1500,
    ).choices[0].message.content

def main():
    Path("eval/data").mkdir(parents=True, exist_ok=True)
    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("langchain")
    r = col.get(include=["documents","metadatas"])
    faqs = []
    for cid, doc, meta in zip(r["ids"], r["documents"], r["metadatas"]):
        m = re.match(r"问题：(.+?)\n答案：(.+)", doc, re.DOTALL)
        if m:
            faqs.append({"chunk_id":cid, "category":meta["category"],
                         "q":m.group(1).strip(), "a":m.group(2).strip()})
    print(f"[1/3] 读取 {len(faqs)} 条 FAQ")

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    samples, qid = [], 0

    print(f"[2/3] 改写中...")
    for i, f in enumerate(faqs):
        out = call_llm(client, REWRITE_PROMPT.format(q=f["q"], a=f["a"]))
        d = extract_json(out)
        if not d or "rewrites" not in d:
            print(f"  ⚠ FAQ {i} 解析失败: {out[:120]}")
            continue
        for rw in d["rewrites"]:
            qid += 1
            samples.append({
                "qid": f"q{qid:03d}",
                "query": rw["query"].strip(),
                "gold_chunk_id": f["chunk_id"],
                "gold_question": f["q"], "gold_answer": f["a"],
                "rewrite_type": rw["type"], "is_answerable": True,
                "category": f["category"],
            })
        print(f"  [{i+1:2d}/{len(faqs)}] {f['q']} → +{len(d['rewrites'])}")

    print(f"[3/3] hard negative...")
    out = call_llm(client, HARD_NEG_PROMPT.format(faqs="\n".join(f"- {x['q']}" for x in faqs)), temp=0.8)
    d = extract_json(out)
    if d and "queries" in d:
        for q in d["queries"][:15]:
            qid += 1
            samples.append({"qid":f"q{qid:03d}","query":q.strip(),
                            "gold_chunk_id":None,"gold_question":None,"gold_answer":None,
                            "rewrite_type":"hard_negative","is_answerable":False,"category":None})
        print(f"  +{min(15,len(d['queries']))}")
    else:
        print(f"  ⚠ 失败: {out[:200] if out else None}")

    with open(OUT, "w", encoding="utf-8") as fp:
        for s in samples:
            fp.write(json.dumps(s, ensure_ascii=False) + "\n")
    pos = sum(1 for s in samples if s["is_answerable"])
    print(f"\n✅ {OUT}: {len(samples)} 条 (正 {pos} + 负 {len(samples)-pos})")
    print("样例:")
    for s in samples[:3] + samples[-2:]:
        print(f"  [{s['rewrite_type']:14s}] {s['query']}")

if __name__ == "__main__":
    main()
