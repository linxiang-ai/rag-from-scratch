"""
RAG Demo v2: Shopee 商品问答检索系统
v2: embedding 召回 Top-20 → cross-encoder reranker 精排 Top-3

环境要求：
- Python 3.10+
- PyTorch 2.7 + CUDA 12.8
- 依赖见 requirements.txt
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch


# ==========================================================
# Step 1: 准备示例数据
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
# Step 2: 转换为 LangChain Document
# ==========================================================
documents = []
for item in faq_data:
    content = f"问题：{item['q']}\n答案：{item['a']}"
    metadata = {"category": item['category'], "question": item['q']}
    documents.append(Document(page_content=content, metadata=metadata))
print(f"[Step 2] 加载 {len(documents)} 条 FAQ 数据")


# ==========================================================
# Step 3: 文档切分
# ==========================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "，", " "]
)
splits = text_splitter.split_documents(documents)
print(f"[Step 3] 文档切分为 {len(splits)} 个 chunks")


# ==========================================================
# Step 4: 加载 Embedding 模型
# ==========================================================
print("[Step 4] 加载 BGE-small-zh-v1.5 ...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
print("[Step 4] Embedding 加载完成")


# ==========================================================
# Step 5: 向量化存储
# ==========================================================
print("[Step 5] 构建 ChromaDB ...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory='./chroma_db'
)
print(f"[Step 5] 向量库构建完成，共 {len(splits)} 个向量")


# ==========================================================
# Step 6: 加载 Reranker (v2 新增)
# ==========================================================
print("[Step 6] 加载 BGE-reranker-base ...")
reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512, device='cuda', default_activation_function=torch.nn.Sigmoid())
print("[Step 6] Reranker 加载完成\n")


# ==========================================================
# Step 7: 两阶段检索 (v2 升级)
# ==========================================================
def search(query: str, recall_k: int = 20, rerank_k: int = 3):
    """embedding 召回 Top-recall_k → reranker 精排 Top-rerank_k"""
    print(f"\nQuery: {query}")
    print("=" * 60)

    candidates = vectorstore.similarity_search_with_score(query, k=recall_k)
    if not candidates:
        print("无召回结果")
        return

    pairs = [(query, doc.page_content) for doc, _ in candidates]
    scores = reranker.predict(pairs).tolist()

    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:rerank_k]

    for i, ((doc, embed_score), rerank_score) in enumerate(reranked, 1):
        print(f"\n[Top {i}] Rerank: {rerank_score:.4f} | Embed: {1 - embed_score:.4f}")
        print(f"分类: {doc.metadata['category']}")
        print(f"内容: {doc.page_content}")
    print("\n" + "=" * 60)


# ==========================================================
# Step 8: 测试 query
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
        search(q)
