# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - 知识库构建（索引入库）

功能：解析文档/图片/视频，生成embedding，保存到本地文件
"""
import os
import base64
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from docx import Document as DocxDocument

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

DOCS_DIR = "disney_knowledge_base"
IMG_DIR = os.path.join(DOCS_DIR, "images")
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"

# 输出文件
INDEX_FILE = "disney_index.faiss"
METADATA_FILE = "disney_metadata.json"

# 切分参数
CHUNK_SIZE = 500  # 每个chunk的字符数
CHUNK_OVERLAP = 50  # chunk之间的重叠字符数

# 视频知识库
VIDEO_KNOWLEDGE = [
    {
        "url": "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4",
        "description": "汽车剐蹭视频"
    }
]


def parse_docx(file_path):
    """解析 DOCX 文件，提取全部文本"""
    doc = DocxDocument(file_path)
    all_text = []

    for element in doc.element.body:
        if element.tag.endswith('p'):
            paragraph_text = ""
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        elif element.tag.endswith('tbl'):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|"*len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n".join(all_text)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按固定长度切分文本"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def get_text_embedding(text):
    """文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_image_embedding(image_path):
    """图片embedding"""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    image_data = f"data:image/{ext};base64,{base64_image}"

    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'image': image_data}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_video_embedding(video_url):
    """视频embedding（多帧取平均）"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'video': video_url}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.message}")

    embeddings = resp.output['embeddings']
    if len(embeddings) > 1:
        vectors = [np.array(e['embedding']) for e in embeddings]
        return np.mean(vectors, axis=0).tolist()
    return embeddings[0]['embedding']


def build_and_save():
    """构建知识库并保存"""
    print("\n--- 构建多模态知识库 ---")
    print(f"切分参数: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    metadata_store = []
    all_vectors = []
    doc_id = 0

    # 处理Word文档
    for filename in os.listdir(DOCS_DIR):
        if filename.startswith('.') or os.path.isdir(os.path.join(DOCS_DIR, filename)):
            continue

        file_path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".docx"):
            print(f"  处理文档: {filename}")
            full_text = parse_docx(file_path)
            chunks = split_text(full_text)
            print(f"    文档长度: {len(full_text)} 字符, 切分为 {len(chunks)} 个chunk")

            for chunk in chunks:
                metadata = {
                    "id": doc_id,
                    "source": filename,
                    "type": "text",
                    "content": chunk
                }

                vector = get_text_embedding(chunk)
                all_vectors.append(vector)
                metadata_store.append(metadata)
                doc_id += 1

    # 处理图片（不再需要OCR，多模态embedding已包含图片语义）
    print("  处理图片...")
    for img_filename in os.listdir(IMG_DIR):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(IMG_DIR, img_filename)
            print(f"    - {img_filename}")

            metadata = {
                "id": doc_id,
                "source": f"图片: {img_filename}",
                "type": "image",
                "path": img_path,
                "content": f"[图片] {img_filename}"
            }

            vector = get_image_embedding(img_path)
            all_vectors.append(vector)
            metadata_store.append(metadata)
            doc_id += 1

    # 处理视频
    print("  处理视频...")
    for video_info in VIDEO_KNOWLEDGE:
        print(f"    - {video_info['description']}")

        metadata = {
            "id": doc_id,
            "source": f"视频: {video_info['description']}",
            "type": "video",
            "url": video_info["url"],
            "description": video_info["description"],
            "content": f"[视频] {video_info['description']}"
        }

        vector = get_video_embedding(video_info["url"])
        all_vectors.append(vector)
        metadata_store.append(metadata)
        doc_id += 1

    # 创建FAISS索引
    if all_vectors:
        dim = len(all_vectors[0])
        print(f"\n向量维度: {dim}")

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_vectors).astype('float32'))

        # 保存索引
        faiss.write_index(index, INDEX_FILE)
        print(f"索引已保存: {INDEX_FILE}")

        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=2)
        print(f"元数据已保存: {METADATA_FILE}")

    # 统计
    text_count = sum(1 for m in metadata_store if m["type"] == "text")
    image_count = sum(1 for m in metadata_store if m["type"] == "image")
    video_count = sum(1 for m in metadata_store if m["type"] == "video")
    print(f"\n完成! 文本:{text_count}, 图片:{image_count}, 视频:{video_count}")

    # 打印所有知识条目
    print("\n--- 知识库内容 ---")
    for m in metadata_store:
        print(f"\n[{m['id']:2d}] [{m['type']:5s}] {m.get('source', '')}")
        print(f"    {m['content']}")


if __name__ == "__main__":
    build_and_save()
