# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - Gradio 前端界面
运行: python 6-disney_app.py
"""
import os
import re
import json
import base64
import numpy as np
import faiss
import jieba
import dashscope
from http import HTTPStatus
from openai import OpenAI
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor, as_completed
import gradio as gr

# ─── 配置 ───────────────────────────────────────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

dashscope.api_key = DASHSCOPE_API_KEY
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MULTIMODAL_EMBEDDING_MODEL = "qwen3-vl-embedding"

# 新知识库目录：由 7-disney_build_full_index.py 生成，包含全量索引和5个分类子索引
# 旧的单文件索引（disney_index.faiss / disney_metadata.json）已废弃
INDEX_DIR = "disney_indexes"
FULL_INDEX_FILE = os.path.join(INDEX_DIR, "disney_full_index.faiss")
FULL_METADATA_FILE = os.path.join(INDEX_DIR, "disney_full_metadata.json")

# 每个分类对应一对独立的 FAISS 索引，用于精确范围检索；"全部"指向合并的全量索引
CATEGORY_INDEXES = {
    "全部":         (FULL_INDEX_FILE, FULL_METADATA_FILE),
    "产品与服务详情":     (os.path.join(INDEX_DIR, "cat1_products_index.faiss"),   os.path.join(INDEX_DIR, "cat1_products_metadata.json")),
    "运营流程与标准作业程序": (os.path.join(INDEX_DIR, "cat2_operations_index.faiss"), os.path.join(INDEX_DIR, "cat2_operations_metadata.json")),
    "特殊情况与应急预案":   (os.path.join(INDEX_DIR, "cat3_emergency_index.faiss"), os.path.join(INDEX_DIR, "cat3_emergency_metadata.json")),
    "客户关系与支持话术":   (os.path.join(INDEX_DIR, "cat4_customer_index.faiss"),  os.path.join(INDEX_DIR, "cat4_customer_metadata.json")),
    "内部知识与工具":     (os.path.join(INDEX_DIR, "cat5_internal_index.faiss"),  os.path.join(INDEX_DIR, "cat5_internal_metadata.json")),
}
RRF_K          = 60   # RRF 常数，控制排名影响的衰减幅度
RETRIEVAL_TOP_K = 10   # 每路检索（文字向量 / 图像向量 / BM25）各取 top-k
RRF_TOP_K       = 15   # RRF 融合后保留 top-k 进入重排序
RERANK_TOP_K    = 5   # Qwen3-VL 重排序后最终保留 top-k
VL_RERANK_MODEL = "qwen-vl-max"  # Qwen3-VL 重排序模型（可改为 qwen3-vl-72b-instruct）

# 自动分类路由规则：按特征关键词匹配，命中数最多的分类胜出，无命中则使用全量索引
# 顺序不影响结果，关键词尽量选取分类内独有的、区分度高的词
_ROUTE_RULES = [
    ("特殊情况与应急预案", [
        "紧急", "走失", "急救", "医疗", "台风", "闭园", "天气", "预警",
        "应急", "救护", "无障碍", "轮椅", "安全事故",
        "emergency", "lost child", "medical", "typhoon", "closure",
    ]),
    ("内部知识与工具", [
        "CRM", "工单", "员工", "培训", "岗位", "导览", "手册",
        "演职", "内部", "P1", "P2", "Salesforce", "操作手册",
        "staff", "employee", "training", "manual",
    ]),
    ("客户关系与支持话术", [
        "话术", "奇迹时刻", "补偿", "安抚", "沟通",
        "常见问题", "禁用词", "HEARD",
        "magic moment", "script",
    ]),
    ("运营流程与标准作业程序", [
        "退款", "退票", "改期", "预订", "订单", "客诉",
        "投诉处理", "升级", "酒店预订", "官方订票",
        "refund", "booking", "complaint",
    ]),
    ("产品与服务详情", [
        "门票", "票价", "年票", "餐饮", "餐厅", "游乐", "设施",
        "身高", "邮轮", "酒店", "会员", "尊享卡", "礼宾",
        "价格", "开放时间", "巡游", "烟花", "地图",
        "ticket", "price", "restaurant", "cruise", "hotel", "annual pass",
    ]),
]


def auto_detect_category(query: str) -> str:
    """根据查询关键词自动匹配最相关的分类索引，无命中时回退全量索引"""
    q = query.lower()
    scores = {cat: sum(1 for kw in kws if kw.lower() in q)
              for cat, kws in _ROUTE_RULES}
    best_cat, best_score = max(scores.items(), key=lambda x: x[1])
    detected = best_cat if best_score > 0 else "全部"
    print(f"[路由] 查询='{query[:30]}' → 分类='{detected}' (得分={best_score})")
    return detected


# ─── 核心 RAG 函数 ──────────────────────────────────────────────────────────
def load_index(index_file=None, metadata_file=None):
    if index_file is None:
        index_file = FULL_INDEX_FILE
    if metadata_file is None:
        metadata_file = FULL_METADATA_FILE
    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        return None, None
    idx = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"知识库已加载: {idx.ntotal} 条记录  [{index_file}]")
    return idx, meta


# 按需缓存：全量索引在启动时预加载，分类索引首次使用时才加载，避免一次性占用过多内存
_index_cache: dict = {}

def get_index(category: str):
    if category not in _index_cache:
        idx_file, meta_file = CATEGORY_INDEXES[category]
        _index_cache[category] = load_index(idx_file, meta_file)
    return _index_cache[category]


def _tokenize(text: str) -> list:
    """jieba 分词 + 英文/数字提取，过滤空白与纯标点"""
    return [t for t in jieba.lcut(text.lower())
            if t.strip() and not re.fullmatch(r'\W+', t)]


_bm25_cache: dict = {}


def _build_bm25(metadata: list) -> BM25Okapi:
    return BM25Okapi([_tokenize(m["content"]) for m in metadata])


def get_bm25(category: str) -> BM25Okapi:
    if category not in _bm25_cache:
        _, meta = get_index(category)
        _bm25_cache[category] = _build_bm25(meta) if meta is not None else None
    return _bm25_cache[category]


def get_text_embedding(text):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding 失败: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def detect_language(text):
    """通过中文字符占比判断语言：中文字符 > 20% 视为中文，否则视为英文"""
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    return "zh" if len(text) > 0 and chinese / len(text) > 0.2 else "en"



# ─── 检索阶段：三路独立通道 ─────────────────────────────────────────────────

def _vector_channels(query_vec: np.ndarray, index, metadata: list,
                     k: int = RETRIEVAL_TOP_K):
    """一次 FAISS 全量搜索，按类型拆成文字和图像两路，各取 top-k"""
    dists, idxs = index.search(query_vec, index.ntotal)
    text_ch, image_ch = [], []
    for idx, dist in zip(idxs[0], dists[0]):
        if idx == -1:
            continue
        r = {
            "doc_id": int(idx),
            "distance": float(dist),
            "similarity": 1.0 / (1.0 + float(dist)),
            "bm25_score": 0.0,
            "metadata": metadata[idx],
        }
        mtype = metadata[idx]["type"]
        if mtype == "text" and len(text_ch) < k:
            text_ch.append(r)
        elif mtype == "image" and len(image_ch) < k:
            image_ch.append(r)
        if len(text_ch) >= k and len(image_ch) >= k:
            break
    return text_ch, image_ch


def _bm25_channel(query: str, bm25: BM25Okapi, metadata: list,
                  k: int = RETRIEVAL_TOP_K) -> list:
    """BM25 关键词检索，只保留 score > 0 的结果，最多取 top-k"""
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    results = []
    for idx in np.argsort(-scores):
        if len(results) >= k:
            break
        if scores[idx] > 0:
            results.append({
                "doc_id": int(idx),
                "distance": float("inf"),
                "similarity": 0.0,
                "bm25_score": float(scores[idx]),
                "metadata": metadata[idx],
            })
    return results


# ─── 融合阶段：RRF ───────────────────────────────────────────────────────────

def _rrf_merge(channels: list, top_k: int = RRF_TOP_K) -> list:
    """将多路检索结果通过 RRF 融合，返回按 rrf_score 降序的 top-k 列表"""
    doc_rrf: dict = {}
    doc_data: dict = {}
    for ch in channels:
        for rank, r in enumerate(ch):
            did = r["doc_id"]
            doc_rrf[did] = doc_rrf.get(did, 0.0) + 1.0 / (RRF_K + rank)
            if did not in doc_data:
                doc_data[did] = r
    sorted_ids = sorted(doc_rrf, key=lambda x: -doc_rrf[x])[:top_k]
    return [{**doc_data[did], "rrf_score": doc_rrf[did], "rerank_score": 0.0}
            for did in sorted_ids]


# ─── 重排序阶段：Qwen3-VL 深度语义打分 ──────────────────────────────────────

def _score_text(query: str, content: str) -> float:
    """用 Qwen3-VL 对文本候选与查询的相关性打分（0-10）"""
    try:
        resp = client.chat.completions.create(
            model=VL_RERANK_MODEL,
            messages=[
                {"role": "system",
                 "content": "你是相关性评分专家。严格只输出0-10之间的一个整数，不要任何其他文字。"},
                {"role": "user",
                 "content": f"查询：{query}\n\n文档（节选）：{content[:600]}\n\n相关性分数："}
            ],
            max_tokens=4,
        )
        m = re.search(r'\d+', resp.choices[0].message.content.strip())
        return min(10.0, float(m.group())) if m else 5.0
    except Exception:
        return 5.0


def _score_image(query: str, image_path: str) -> float:
    """用 Qwen3-VL 对图片候选与查询的相关性打分（0-10）"""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png"}.get(ext, "image/jpeg")
        resp = client.chat.completions.create(
            model=VL_RERANK_MODEL,
            messages=[
                {"role": "system",
                 "content": "你是相关性评分专家。严格只输出0-10之间的一个整数，不要任何其他文字。"},
                {"role": "user",
                 "content": [
                     {"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64,{b64}"}},
                     {"type": "text",
                      "text": f"查询：{query}\n\n上图与查询的相关性分数（0-10整数）："}
                 ]}
            ],
            max_tokens=4,
        )
        m = re.search(r'\d+', resp.choices[0].message.content.strip())
        return min(10.0, float(m.group())) if m else 5.0
    except Exception:
        return 5.0


def _score_relevance(query: str, result: dict) -> float:
    """根据结果类型分发到对应的打分函数"""
    if result["metadata"]["type"] == "image":
        raw = result["metadata"].get("path", "")
        abs_path = os.path.abspath(raw) if raw else ""
        if abs_path and os.path.exists(abs_path):
            return _score_image(query, abs_path)
    return _score_text(query, result["metadata"].get("content", ""))


def _rerank_with_vl(query: str, candidates: list,
                    top_k: int = RERANK_TOP_K) -> list:
    """并发调用 Qwen3-VL 对 RRF top-k 候选深度重排序，返回最终 top-k"""
    scores: dict = {}
    with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as pool:
        futures = {pool.submit(_score_relevance, query, r): i
                   for i, r in enumerate(candidates)}
        for fut in as_completed(futures):
            i = futures[fut]
            scores[i] = fut.result() if not fut.exception() else 5.0

    scored = [{**candidates[i], "rerank_score": scores.get(i, 5.0)}
              for i in range(len(candidates))]
    scored.sort(key=lambda x: -x["rerank_score"])
    print("[重排序] " + " | ".join(
        f"{r['metadata']['source'][:15]}={r['rerank_score']:.0f}" for r in scored))
    return scored[:top_k]


def rag_query(query, index, metadata, k=RERANK_TOP_K, category="全部"):
    # ── 阶段一：内容召回（三路独立，各取 top-5） ─────────────────────────────
    query_vec = np.array([get_text_embedding(query)]).astype("float32")
    bm25      = get_bm25(category)

    text_ch, image_ch = _vector_channels(query_vec, index, metadata, k=RETRIEVAL_TOP_K)
    bm25_ch           = _bm25_channel(query, bm25, metadata, k=RETRIEVAL_TOP_K)
    print(f"[召回] 文字向量:{len(text_ch)}  图像向量:{len(image_ch)}  BM25:{len(bm25_ch)}")

    # ── 阶段二：RRF 融合三路结果 → top-7 ─────────────────────────────────────
    fused = _rrf_merge([text_ch, image_ch, bm25_ch], top_k=RRF_TOP_K)
    print(f"[RRF] top-{len(fused)}: {[r['metadata']['source'][:18] for r in fused]}")

    # ── 阶段三：Qwen3-VL 深度语义重排序 → top-3 ──────────────────────────────
    reranked = _rerank_with_vl(query, fused, top_k=k)

    # ── 阶段四：生成回答 ──────────────────────────────────────────────────────
    top_texts     = [r for r in reranked if r["metadata"]["type"] == "text"]
    matched_image = next((r for r in reranked if r["metadata"]["type"] == "image"), None)
    matched_video = next((r for r in reranked if r["metadata"]["type"] == "video"), None)

    context_str = "".join(
        f"背景知识 {i+1} (来源: {r['metadata']['source']}, 重排得分: {r['rerank_score']:.1f}/10):\n"
        f"{r['metadata']['content']}\n\n"
        for i, r in enumerate(top_texts)
    )

    media_hint = ""
    if matched_image:
        media_hint += "\n\n[系统提示：已检索到相关图片，将在界面右侧展示，请在回答中引导用户查看右侧图片区域，不要说无法提供图片。]"
    if matched_video:
        media_hint += "\n\n[系统提示：已检索到相关视频链接，将在界面右侧提供，请在回答中引导用户查看右侧视频区域。]"

    lang = detect_language(query)
    if lang == "en":
        system_prompt = "You are a helpful Disney customer service assistant. Be concise and friendly. You MUST reply in English regardless of the language of the background knowledge."
        user_prompt   = f"Answer the user's question based on the following background knowledge.\n\n[Background Knowledge]\n{context_str}\n[User Question]\n{query}{media_hint}"
    else:
        system_prompt = "你是一个专业友好的迪士尼客服助手，回答简洁清晰。必须用中文回答。"
        user_prompt   = f"请根据以下背景知识回答用户问题。\n\n[背景知识]\n{context_str}\n[用户问题]\n{query}{media_hint}"

    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )

    image_path = None
    if matched_image:
        raw = matched_image["metadata"].get("path", "")
        image_path = os.path.abspath(raw) if raw else None

    return {
        "answer":    completion.choices[0].message.content,
        "image_path": image_path,
        "video_url": matched_video["metadata"].get("url") if matched_video else None,
        "sources": [
            {
                "category":     r["metadata"].get("category", "-"),
                "source":       r["metadata"]["source"],
                "rerank_score": r["rerank_score"],
                "rrf_score":    r["rrf_score"],
                "content":      r["metadata"]["content"][:100],
            }
            for r in reranked
        ]
    }


# ─── 预加载全量索引 + BM25 ───────────────────────────────────────────────────
_index_cache["全部"] = load_index(FULL_INDEX_FILE, FULL_METADATA_FILE)
_, _preload_meta = _index_cache["全部"]
if _preload_meta is not None:
    _bm25_cache["全部"] = _build_bm25(_preload_meta)
    print(f"BM25 已预建: 全部 ({len(_preload_meta)} 条记录)")


# ─── Gradio 事件处理 ─────────────────────────────────────────────────────────
def on_user_submit(message, history):
    """将用户消息追加到对话历史，清空输入框，同时把原始字符串存入 State"""
    history = history or []
    history.append({"role": "user", "content": message})
    return "", history, message          # 第三个返回值写入 last_query State


def on_bot_respond(last_query, history):
    """用 State 里保存的原始字符串执行 RAG，避免 Gradio 序列化破坏 content 类型"""
    query = last_query
    if not query or not history:
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    category = auto_detect_category(query)
    faiss_index, faiss_metadata = get_index(category)
    if faiss_index is None:
        history.append({"role": "assistant", "content": "⚠️ 知识库尚未构建，请先运行 `7-disney_build_full_index.py` 生成索引文件。"})
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    try:
        result = rag_query(query, faiss_index, faiss_metadata, category=category)
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ 查询出错: {e}"})
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    history.append({"role": "assistant", "content": result["answer"]})

    img_update = (
        gr.update(value=result["image_path"], visible=True)
        if result["image_path"]
        else gr.update(visible=False)
    )

    if result["video_url"]:
        url = result["video_url"]
        video_html = (
            '<div style="padding:10px;border:1px solid #e0e0e0;border-radius:8px;margin-top:4px">'
            f'<b>相关视频</b><br>'
            f'<a href="{url}" target="_blank" style="color:#1a73e8;word-break:break-all">{url}</a>'
            '</div>'
        )
        video_update = gr.update(value=video_html, visible=True)
    else:
        video_update = gr.update(value="", visible=False)

    sources_md = "| 分类 | 来源文件 | 重排得分 | RRF分 | 内容摘要 |\n|------|---------|---------|-------|----------|\n"
    for s in result["sources"]:
        sources_md += (f"| {s.get('category', '-')} | {s['source']} "
                       f"| {s['rerank_score']:.1f}/10 | {s['rrf_score']:.4f} "
                       f"| {s['content']}... |\n")

    return history, img_update, video_update, sources_md


def on_clear():
    return [], gr.update(visible=False), gr.update(value="", visible=False), ""  # [] = 空 messages 列表


# ─── UI 布局 ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="迪士尼AI客服助手") as demo:

    gr.Markdown(
        "# 🏰 迪士尼AI客服助手\n"
        "> 基于内部文档的多模态RAG问答系统，支持文本、图片、视频检索"
    )

    with gr.Row():
        # ── 对话区 ──────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话",
                height=500,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="例如：迪士尼门票的退款流程是什么？/ 万圣节的活动海报",
                    show_label=False,
                    scale=5,
                    max_lines=3,
                )
                send_btn = gr.Button("发送", scale=1, variant="primary")
            clear_btn = gr.Button("清空对话", size="sm")

        # ── 媒体 & 来源面板 ──────────────────────────────────────────
        with gr.Column(scale=2):
            result_image = gr.Image(
                label="相关图片",
                visible=False,
                height=260,
            )
            result_video = gr.HTML(visible=False)
            with gr.Accordion("检索来源", open=False):
                sources_md = gr.Markdown("")

    gr.Examples(
        examples=[
            "迪士尼门票的退款流程是什么？",
            "老人票有哪些优惠规定？",
            "上海迪士尼酒店会员有什么专属福利？",
            "最近万圣节的活动海报是什么样的？",
            "我的汽车被剐蹭了，有相关视频可以看吗？",
        ],
        inputs=msg_box,
        label="示例问题（点击填入）",
    )

    last_query = gr.State("")   # 保存未经 Gradio 序列化的原始用户输入

    # ── 事件绑定 ─────────────────────────────────────────────────────
    _outputs = [chatbot, result_image, result_video, sources_md]

    send_btn.click(
        on_user_submit, [msg_box, chatbot], [msg_box, chatbot, last_query]
    ).then(
        on_bot_respond, [last_query, chatbot], _outputs
    )

    msg_box.submit(
        on_user_submit, [msg_box, chatbot], [msg_box, chatbot, last_query]
    ).then(
        on_bot_respond, [last_query, chatbot], _outputs
    )

    clear_btn.click(on_clear, outputs=_outputs)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=9001,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
