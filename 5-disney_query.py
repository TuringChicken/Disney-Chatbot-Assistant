# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 V2 - 查询处理

功能：加载索引，处理用户query，打印相似度排名，支持图片/视频关键词检测
"""
import os
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from openai import OpenAI

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"
INDEX_FILE = "disney_index.faiss"
METADATA_FILE = "disney_metadata.json"

# 关键词配置
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]
MEDIA_DISTANCE_THRESHOLD = 3.0  # 图片/视频匹配的距离阈值


def load_index():
    """加载索引和元数据"""
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"已加载索引: {index.ntotal} 条记录")
    return index, metadata


def get_text_embedding(text):
    """文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def distance_to_similarity(distance):
    """L2距离转相似度 (0-1之间，越大越相似)"""
    return 1 / (1 + distance)


def detect_media_intent(query):
    """检测query中是否包含图片/视频意图"""
    query_lower = query.lower()
    want_image = any(kw in query_lower for kw in IMAGE_KEYWORDS)
    want_video = any(kw in query_lower for kw in VIDEO_KEYWORDS)
    return want_image, want_video


def search_with_details(query, index, metadata):
    """检索并打印相似度详情"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    query_vec = np.array([get_text_embedding(query)]).astype('float32')

    # 检索所有
    distances, indices = index.search(query_vec, index.ntotal)

    print(f"\n相似度排名 (越大越相似):")
    print("-" * 80)
    print(f"{'排名':4s} {'ID':4s} {'类型':6s} {'相似度':8s} {'距离':8s} 内容")
    print("-" * 80)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        m = metadata[idx]
        sim = distance_to_similarity(dist)
        content_preview = m['content'][:45].replace('\n', ' ')
        type_tag = m['type']

        marker = ""
        if type_tag == "image":
            marker = " <-- 图片"
        elif type_tag == "video":
            marker = " <-- 视频"

        print(f"{rank+1:4d} {idx:4d} [{type_tag:5s}] {sim:6.4f}  {dist:8.4f}  {content_preview}...{marker}")
        results.append({"idx": idx, "distance": dist, "similarity": sim, "metadata": m})

    return results


def rag_ask(query, index, metadata, k=3):
    """RAG问答，支持图片/视频关键词检测"""
    results = search_with_details(query, index, metadata)

    # 检测媒体意图
    want_image, want_video = detect_media_intent(query)
    print(f"\n意图检测: 需要图片={want_image}, 需要视频={want_video}")

    # 取top-k文本结果
    top_results = [r for r in results if r["metadata"]["type"] == "text"][:k]

    # 如果需要图片，找距离<3的图片中距离最小的Top1
    matched_image = None
    if want_image:
        image_results = [r for r in results if r["metadata"]["type"] == "image" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if image_results:
            # 按距离排序，取距离最小的Top1
            image_results.sort(key=lambda x: x["distance"])
            matched_image = image_results[0]
            print(f"  -> 匹配到图片: {matched_image['metadata']['path']} (距离: {matched_image['distance']:.4f}, 相似度: {matched_image['similarity']:.4f})")

    # 如果需要视频，找距离<3的视频中距离最小的Top1
    matched_video = None
    if want_video:
        video_results = [r for r in results if r["metadata"]["type"] == "video" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if video_results:
            # 按距离排序，取距离最小的Top1
            video_results.sort(key=lambda x: x["distance"])
            matched_video = video_results[0]
            print(f"  -> 匹配到视频: {matched_video['metadata']['url']} (距离: {matched_video['distance']:.4f}, 相似度: {matched_video['similarity']:.4f})")

    print(f"\n选取Top-{k}文本构建Prompt:")
    for r in top_results:
        print(f"  - {r['metadata']['content'][:50]}... (相似度: {r['similarity']:.4f})")

    # 构建context
    context_str = ""
    for i, r in enumerate(top_results):
        m = r["metadata"]
        context_str += f"背景知识 {i+1} (来源: {m['source']}, 相似度: {r['similarity']:.4f}):\n{m['content']}\n\n"

    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识回答用户问题。

[背景知识]
{context_str}
[用户问题]
{query}
"""

    # 调用LLM
    print("\n调用LLM生成答案...")
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content": "你是一个迪士尼客服助手。"},
            {"role": "user", "content": prompt}
        ]
    )
    answer = completion.choices[0].message.content

    # 附加匹配到的媒体
    if matched_image:
        answer += f"\n\n[相关图片]: {matched_image['metadata']['path']}"
    if matched_video:
        answer += f"\n\n[相关视频]: {matched_video['metadata']['url']}"

    print(f"\n最终答案:\n{answer}")
    return answer


if __name__ == "__main__":
    index, metadata = load_index()

    print("\n" + "="*60)
    print("测试1: 文本查询")
    rag_ask("我想了解一下迪士尼门票的退款流程", index, metadata, k=3)

    print("\n" + "="*60)
    print("测试2: 图片查询 - 万圣节海报")
    rag_ask("最近万圣节的活动海报是什么", index, metadata, k=3)

    print("\n" + "="*60)
    print("测试3: 视频查询 - 汽车剐蹭")
    rag_ask("我的汽车被剐蹭了，你能看到视频么？", index, metadata, k=3)

    print("\n" + "="*60)
    print("测试4: 图片查询 - 聚在一起")
    rag_ask("聚在一起说奇妙的海报", index, metadata, k=3)
