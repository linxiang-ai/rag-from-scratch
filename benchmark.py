"""
vLLM 推理性能 benchmark
- Test 1: 单 query 延迟（warmup 后 N 次取统计）
- Test 2: 并发吞吐（N 个 query 同时发起，看 QPS）
"""
import asyncio
import time
import statistics
from openai import OpenAI, AsyncOpenAI

LLM_BASE_URL = "http://localhost:8000/v1"
LLM_NAME = "qwen2.5-7b"
MAX_TOKENS = 100

TEST_QUERIES = [
    "你好，请简要介绍一下自己",
    "什么是 RAG？",
    "PyTorch 和 TensorFlow 有什么区别？",
    "解释一下注意力机制",
    "用一句话总结大语言模型的核心",
]


def single_query(client, query):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_NAME,
        messages=[{"role": "user", "content": query}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return time.time() - t0, resp.usage.completion_tokens


def bench_latency(n_runs=5):
    client = OpenAI(base_url=LLM_BASE_URL, api_key="dummy")

    print("=" * 60)
    print(f"[Test 1] Single-query Latency (warmup + {n_runs} runs/query, {len(TEST_QUERIES)} queries)")
    print("=" * 60)

    print("Warming up ...")
    for q in TEST_QUERIES[:2]:
        single_query(client, q)

    latencies, tokens_list, tps_list = [], [], []
    for q in TEST_QUERIES:
        for _ in range(n_runs):
            elapsed, n_tokens = single_query(client, q)
            latencies.append(elapsed)
            tokens_list.append(n_tokens)
            tps_list.append(n_tokens / elapsed)

    print(f"  Avg latency:     {statistics.mean(latencies)*1000:.1f} ms")
    print(f"  Median latency:  {statistics.median(latencies)*1000:.1f} ms")
    print(f"  P95 latency:     {sorted(latencies)[int(len(latencies)*0.95)]*1000:.1f} ms")
    print(f"  Avg tokens/sec:  {statistics.mean(tps_list):.1f}")
    print(f"  Avg tokens:      {statistics.mean(tokens_list):.1f}")


async def async_query(client, query):
    t0 = time.time()
    resp = await client.chat.completions.create(
        model=LLM_NAME,
        messages=[{"role": "user", "content": query}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return time.time() - t0, resp.usage.completion_tokens


async def bench_throughput(n_concurrent=10):
    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="dummy")

    print()
    print("=" * 60)
    print(f"[Test 2] Concurrent Throughput ({n_concurrent} queries in parallel)")
    print("=" * 60)

    queries = (TEST_QUERIES * ((n_concurrent // len(TEST_QUERIES)) + 1))[:n_concurrent]

    t0 = time.time()
    results = await asyncio.gather(*[async_query(client, q) for q in queries])
    wall_time = time.time() - t0

    total_tokens = sum(r[1] for r in results)
    qps = n_concurrent / wall_time
    aggregate_tps = total_tokens / wall_time

    print(f"  Wall time:        {wall_time*1000:.1f} ms")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  QPS:              {qps:.2f}")
    print(f"  Throughput (agg): {aggregate_tps:.1f} tokens/sec")


if __name__ == "__main__":
    bench_latency(n_runs=5)
    asyncio.run(bench_throughput(n_concurrent=10))
