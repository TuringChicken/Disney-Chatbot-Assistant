#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dashscope
import base64
import json
from http import HTTPStatus
# 读取图片并转换为Base64
image_path = "CASE-迪士尼RAG助手/disney_knowledge_base/images/1-聚在一起说奇妙.jpg"
with open(image_path, "rb") as image_file:
    # 读取文件并转换为Base64
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
# 设置图像格式
image_format = "jpg"  # 根据实际情况修改，比如jpg,bmp,png 等
image_data = f"data:image/{image_format};base64,{base64_image}"
# 输入数据
input = [{'image': image_data}]

# 调用模型接口
resp = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",
    input=input
)
if resp.status_code == HTTPStatus.OK:
    result = {
        "status_code": resp.status_code,
        "request_id": getattr(resp, "request_id", ""),
        "code": getattr(resp, "code", ""),
        "message": getattr(resp, "message", ""),
        "output": resp.output,
        "usage": resp.usage
    }
    print(json.dumps(result, ensure_ascii=False, indent=4))

