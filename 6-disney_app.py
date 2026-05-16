# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - Gradio 前端界面
运行: python 6-disney_app.py
"""
import os
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from openai import OpenAI
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

MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"

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
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图",
                  "image", "photo", "poster", "picture", "show", "look", "see"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放",
                  "video", "watch", "play", "clip", "footage"]
MEDIA_DISTANCE_THRESHOLD = 3.0

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


def detect_media_intent(query):
    q = query.lower()
    return (
        any(kw in q for kw in IMAGE_KEYWORDS),
        any(kw in q for kw in VIDEO_KEYWORDS)
    )


def rag_query(query, index, metadata, k=3, category="全部"):
    vec = np.array([get_text_embedding(query)]).astype("float32")
    distances, indices = index.search(vec, index.ntotal)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        results.append({
            "distance": float(dist),
            "similarity": 1 / (1 + float(dist)),
            "metadata": metadata[idx]
        })

    want_image, want_video = detect_media_intent(query)
    top_texts = [r for r in results if r["metadata"]["type"] == "text"][:k]

    def best_media(mtype):
        candidates = [r for r in results
                      if r["metadata"]["type"] == mtype
                      and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        return min(candidates, key=lambda x: x["distance"]) if candidates else None

    matched_image = best_media("image") if want_image else None
    matched_video = best_media("video") if want_video else None

    context_str = "".join(
        f"背景知识 {i+1} (来源: {r['metadata']['source']}, 相似度: {r['similarity']:.4f}):\n"
        f"{r['metadata']['content']}\n\n"
        for i, r in enumerate(top_texts)
    )

    # 告知 LLM 已检索到的媒体资源，避免它说"无法展示图片/视频"
    media_hint = ""
    if matched_image:
        media_hint += "\n\n[系统提示：已检索到相关图片，将在界面右侧展示，请在回答中引导用户查看右侧图片区域，不要说无法提供图片。]"
    if matched_video:
        media_hint += "\n\n[系统提示：已检索到相关视频链接，将在界面右侧提供，请在回答中引导用户查看右侧视频区域。]"

    lang = detect_language(query)
    if lang == "en":
        system_prompt = "You are a helpful Disney customer service assistant. Be concise and friendly. You MUST reply in English regardless of the language of the background knowledge."
        user_prompt = f"Answer the user's question based on the following background knowledge.\n\n[Background Knowledge]\n{context_str}\n[User Question]\n{query}{media_hint}"
    else:
        system_prompt = "你是一个专业友好的迪士尼客服助手，回答简洁清晰。必须用中文回答。"
        user_prompt = f"请根据以下背景知识回答用户问题。\n\n[背景知识]\n{context_str}\n[用户问题]\n{query}{media_hint}"

    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 将图片路径转为绝对路径，避免工作目录不同导致图片加载失败
    image_path = None
    if matched_image:
        raw_path = matched_image["metadata"].get("path", "")
        image_path = os.path.abspath(raw_path) if raw_path else None

    return {
        "answer": completion.choices[0].message.content,
        "image_path": image_path,
        "video_url": matched_video["metadata"].get("url") if matched_video else None,
        "sources": [
            {
                "category": r["metadata"].get("category", "-"),
                "source": r["metadata"]["source"],
                "similarity": r["similarity"],
                "content": r["metadata"]["content"][:100]
            }
            for r in top_texts
        ]
    }


# ─── 预加载全量索引 ──────────────────────────────────────────────────────────
_index_cache["全部"] = load_index(FULL_INDEX_FILE, FULL_METADATA_FILE)


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

    sources_md = "| 分类 | 来源文件 | 相似度 | 内容摘要 |\n|------|---------|--------|----------|\n"
    for s in result["sources"]:
        sources_md += f"| {s.get('category', '-')} | {s['source']} | {s['similarity']:.4f} | {s['content']}... |\n"

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
