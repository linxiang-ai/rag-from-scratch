"""
RAG Demo v2.1 (milestone 2a)
完整链路: embedding 召回 Top-20 → reranker 精排 Top-3 → Qwen2.5-7B 生成答案

环境要求：
- Python 3.10+ · PyTorch 2.7 · CUDA 12.8
- 显存约 17 GB（BGE + reranker + Qwen2.5-7B bf16）
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import shutil
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================================
# Step 1: 数据
# ==========================================================
faq_data = [
    {"category": "退款", "q": "怎么申请退款", "a": "在订单详情页点击'申请退款'，选择退款原因后上传凭证，等待商家处理。一般 3-5 个工作日内有反馈。"},
    {"category": "退款", "q": "退款多久能到账", "a": "退款审核通过后，原路返回支付账户，信用卡 1-7 个工作日，钱包账户即时到账。"},
    {"category": "退款", "q": "已发货能退款吗", "a": "已发货商品需先签收，然后申请'退货退款'。拒收的包裹会被退回，签收前不建议直接申请退款。"},
    {"category": "发货", "q": "什么时候发货", "a": "卖家会在买家付款后 48 小时内发货。预售商品按商品页注明的预售期发货。"},
    {"category": "发货", "q": "为什么我的订单还没发货", "a": "可能是订单审核中或库存调拨中。超过 72 小时未发货，可联系客服催单或申请退款。"},
    {"category": "发货", "q": "发货后多久到", "a": "国内一般 3-7 天，跨境物流 7-15 天，具体看物流公司和地址。"},
    {"category": "物流", "q": "物流信息不更新", "a": "物流轨迹更新有延迟是正常的。若超过 3 天无更新，请联系物流公司或在订单页申请客服介入。"},
    {"category": "物流", "q": "包裹丢失怎么办", "a": "凭物流单号联系物流公司核实，若确认丢失可申请赔付。也可联系平台客服协助处理。"},
    {"category": "支付", "q": "支持哪些支付方式", "a": "支持信用卡、借记卡、ShopeePay 钱包、银行转账等。印尼地区还支持 OVO、DANA、GoPay 等本地钱包。"},
    {"category": "支付", "q": "支付失败怎么办", "a": "检查银行卡余额、网络连接、是否超出限额。可尝试更换支付方式或稍后重试。"},
    {"category": "账户", "q": "怎么修改手机号", "a": "进入'我的-设置-账户安全-修改手机号'，按提示验证身份后修改。"},
    {"category": "账户", "q": "账号被冻结了怎么办", "a": "可能因异常登录或违规操作。联系客服提供身份证明即可申诉解冻。"},
    {"category": "优惠", "q": "怎么领取优惠券", "a": "首页'领券中心'或商品详情页都可领取。注意优惠券的使用门槛和有效期。"},
    {"category": "优惠", "q": "优惠券能叠加吗", "a": "店铺优惠券和平台优惠券可以叠加，同类优惠券（如两张店铺券）不能叠加。"},
    {"category": "商品", "q": "商品质量有问题", "a": "7 天无理由退换货，质量问题可申请'仅退款'或'退货退款'，平台会协助处理。"},
    {"category": "商品", "q": "怎么评价商品", "a": "确认收货后 14 天内可评价。在'已完成-订单详情'中点击评价即可。"},
]


# ==========================================================
# Step 2: Document 转换
# ==========================================================
documents = []
for item in faq_data:
    content = f"问题：{item['q']}\n答案：{item['a']}"
    metadata = {"category": item['category'], "question": item['q']}
    documents.append(Document(page_content=content, metadata=metadata))
print(f"[Step 2] 加载 {len(documents)} 条 FAQ")


# ==========================================================
# Step 3: 文档切分
# ==========================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "，", " "]
)
splits = text_splitter.split_documents(documents)
print(f"[Step 3] 切分为 {len(splits)} 个 chunks")


# ==========================================================
# Step 4: Embedding
# ==========================================================
print("[Step 4] 加载 BGE-small-zh-v1.5 ...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)


# ==========================================================
# Step 5: ChromaDB（每次重建避免累积）
# ==========================================================
shutil.rmtree('./chroma_db', ignore_errors=True)
print("[Step 5] 构建 ChromaDB ...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory='./chroma_db'
)


# ==========================================================
# Step 6: Reranker
# ==========================================================
print("[Step 6] 加载 BGE-reranker-base ...")
reranker = CrossEncoder(
    'BAAI/bge-reranker-base',
    max_length=512,
    device='cuda',
    default_activation_function=torch.nn.Sigmoid()
)


# ==========================================================
# Step 7: LLM (Qwen2.5-7B-Instruct)
# ==========================================================
LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
print(f"[Step 7] 加载 {LLM_NAME} ...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
)
llm_model.eval()
print("[Step 7] LLM 加载完成\n")


SYSTEM_PROMPT = """你是 Shopee 电商平台的客服助手。请根据下面提供的 FAQ 上下文回答用户问题：
- 如果上下文里有明确答案，简洁准确地回复（不超过 80 字）
- 如果上下文不足以回答，回复"抱歉，我没有相关信息，建议联系人工客服"
- 不要编造上下文之外的信息或政策"""


# ==========================================================
# Step 8: 检索 + 重排
# ==========================================================
def retrieve_and_rerank(query, recall_k=20, rerank_k=3):
    candidates = vectorstore.similarity_search_with_score(query, k=recall_k)
    if not candidates:
        return []
    pairs = [(query, doc.page_content) for doc, _ in candidates]
    scores = reranker.predict(pairs).tolist()
    reranked = sorted(
        zip([d for d, _ in candidates], scores),
        key=lambda x: x[1],
        reverse=True
    )[:rerank_k]
    return reranked


# ==========================================================
# Step 9: LLM 生成
# ==========================================================
@torch.no_grad()
def generate(query, contexts):
    context_text = "\n\n".join(f"[FAQ {i+1}]\n{c}" for i, c in enumerate(contexts))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context_text}\n\n用户问题：{query}"},
    ]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer(text, return_tensors='pt').to('cuda')
    output_ids = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.05,
    )
    response = llm_tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


# ==========================================================
# Step 10: 完整 RAG 链路
# ==========================================================
def ask(query):
    print(f"\n[Query] {query}")
    print("=" * 60)
    reranked = retrieve_and_rerank(query)
    print(f"[Retrieved Top-{len(reranked)}]")
    for i, (doc, score) in enumerate(reranked, 1):
        preview = doc.page_content.replace('\n', ' ')[:80]
        print(f"  [{i}] {doc.metadata['category']} | Rerank: {score:.4f}")
        print(f"      {preview}...")
    if not reranked or reranked[0][1] < 0.2:
        print(f"\n[Answer]\n抱歉，我没有相关信息，建议联系人工客服。(Top-1 score={reranked[0][1]:.4f} < 0.2)")
        print("=" * 60)
        return

    contexts = [doc.page_content for doc, _ in reranked]
    answer = generate(query, contexts)
    print(f"\n[Answer]\n{answer}")
    print("=" * 60)


# ==========================================================
# Step 11: 测试
# ==========================================================
if __name__ == "__main__":
    test_queries = [
        "我要退货",
        "钱什么时候能退回来",
        "物流停了好几天了",
        "怎么用印尼本地钱包付款",
        "买的东西坏了",
    ]
    for q in test_queries:
        ask(q)
