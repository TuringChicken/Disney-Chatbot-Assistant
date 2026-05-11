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
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图",
                  "image", "photo", "poster", "picture", "show", "look", "see"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放",
                  "video", "watch", "play", "clip", "footage"]
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
    """将用户消息追加到对话历史，清空输入框，同时把原始字符串存入 State"""
    history = history or []
    history.append({"role": "user", "content": message})
    return "", history, message          # 第三个返回值写入 last_query State


def on_bot_respond(last_query, history):
    """用 State 里保存的原始字符串执行 RAG，避免 Gradio 序列化破坏 content 类型"""
    query = last_query
    if not query or not history:
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    if faiss_index is None:
        history.append({"role": "assistant", "content": "⚠️ 知识库尚未构建，请先运行 `4-disney_build_index.py` 生成索引文件。"})
        return history, gr.update(visible=False), gr.update(value="", visible=False), ""

    try:
        result = rag_query(query, faiss_index, faiss_metadata)
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

    sources_md = "| 来源文件 | 相似度 | 内容摘要 |\n|---------|--------|----------|\n"
    for s in result["sources"]:
        sources_md += f"| {s['source']} | {s['similarity']:.4f} | {s['content']}... |\n"

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
        server_port=9000,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
