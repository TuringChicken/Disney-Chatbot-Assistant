#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 多模态向量化模型目前仅支持以URL形式输入视频文件，暂不支持直接传入本地视频。
import dashscope
import json
from http import HTTPStatus
# 实际使用中请将url地址替换为您的视频url地址
video = "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4"
input = [{'video': video}]
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

