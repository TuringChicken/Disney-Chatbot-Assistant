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
INDEX_FILE = "disney_index.faiss"
METADATA_FILE = "disney_metadata.json"
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]
MEDIA_DISTANCE_THRESHOLD = 3.0


# ─── 核心 RAG 函数 ──────────────────────────────────────────────────────────
def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None
    idx = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"知识库已加载: {idx.ntotal} 条记录")
    return idx, meta


def get_text_embedding(text):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding 失败: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def detect_media_intent(query):
    q = query.lower()
    return (
        any(kw in q for kw in IMAGE_KEYWORDS),
        any(kw in q for kw in VIDEO_KEYWORDS)
    )


def rag_query(query, index, metadata, k=3):
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

    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content": "你是一个专业友好的迪士尼客服助手，回答简洁清晰。"},
            {"role": "user", "content":
                f"请根据以下背景知识回答用户问题。\n\n[背景知识]\n{context_str}\n[用户问题]\n{query}"}
        ]
    )

    return {
        "answer": completion.choices[0].message.content,
        "image_path": matched_image["metadata"].get("path") if matched_image else None,
        "video_url": matched_video["metadata"].get("url") if matched_video else None,
        "sources": [
            {
                "source": r["metadata"]["source"],
                "similarity": r["similarity"],
                "content": r["metadata"]["content"][:100]
            }
            for r in top_texts
        ]
    }


# ─── 全局加载索引 ────────────────────────────────────────────────────────────
faiss_index, faiss_metadata = load_index()


# ─── Gradio 事件处理 ─────────────────────────────────────────────────────────
def on_user_submit(message, history):
    """将用户消息追加到对话历史（tuple格式），清空输入框"""
    history = history or []
    history.append((message, None))   # bot_msg 占位 None，等待回复
    return "", history


def on_bot_respond(history):
    """根据最后一条未回复的用户消息执行 RAG 并填入回复"""
    if not history or history[-1][1] is not None:
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    query = history[-1][0]

    if faiss_index is None:
        history[-1] = (query, "⚠️ 知识库尚未构建，请先运行 `4-disney_build_index.py` 生成索引文件。")
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    try:
        result = rag_query(query, faiss_index, faiss_metadata)
    except Exception as e:
        history[-1] = (query, f"❌ 查询出错: {e}")
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    history[-1] = (query, result["answer"])

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

    sources_md = "| 来源文件 | 相似度 | 内容摘要 |\n|---------|--------|----------|\n"
    for s in result["sources"]:
        sources_md += f"| {s['source']} | {s['similarity']:.4f} | {s['content']}... |\n"

    return history, img_update, video_update, sources_md


def on_clear():
    return [], gr.update(visible=False), gr.update(value="", visible=False), ""


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
                show_copy_button=True,
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
                show_download_button=True,
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

    # ── 事件绑定 ─────────────────────────────────────────────────────
    _outputs = [chatbot, result_image, result_video, sources_md]

    send_btn.click(
        on_user_submit, [msg_box, chatbot], [msg_box, chatbot]
    ).then(
        on_bot_respond, [chatbot], _outputs
    )

    msg_box.submit(
        on_user_submit, [msg_box, chatbot], [msg_box, chatbot]
    ).then(
        on_bot_respond, [chatbot], _outputs
    )

    clear_btn.click(on_clear, outputs=_outputs)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
