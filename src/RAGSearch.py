#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@CreateTime    : 2025/2/12 10:29
@Author        : wang yi ming
@file for      : 基于缓存和FAISS索引进行检索
"""

import pickle

import numpy as np
import torch
from datetime import datetime

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModel

from config.local_config import sp_group_name_index, text_model_path, doc_chunks_cache, FinBERT
from logger_control import log_error, log_info


# from middleware.redisClient import RedisClient


# 全局加载嵌入模型和索引、文本块缓存
def load_embed_model():
    # return SentenceTransformer(text_model_path, device='cpu')
    return SentenceTransformer('shibing624/text2vec-bge-large-chinese', device='cpu')


def load_faiss_index(index_name):
    return faiss.read_index(index_name)


def read_doc_chunks_cache():
    with open(doc_chunks_cache, 'rb') as f:
        cached_doc_chunks = pickle.load(f)
    return cached_doc_chunks


# 检索函数，使用全局加载的模型、索引和文档块
def retrieve(query, embed_model, index, doc_chunks, top_k=3):
    # 对查询进行向量化
    log_info.info(f"start 对{query}查询进行向量化: {datetime.now()}")
    query_vec = embed_model.encode([query]).astype('float32')
    log_info.info(f"end 对{query}查询进行向量化: {datetime.now()}")
    # 在 FAISS 索引中搜索最相似的 top_k 个
    distances, indices = index.search(query_vec, top_k)
    # 根据索引获取对应的文本块
    retrieved_chunks = [doc_chunks[i] for i in indices[0]]
    return retrieved_chunks


def retrieve_bert(text, tokenizer, bert_model, index, doc_chunks, top_k=3):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)

    embedding = outputs.pooler_output[0].numpy()
    query_vec = np.array([embedding]).astype('float32')
    # 在 FAISS 索引中搜索最相似的 top_k 个
    distances, indices = index.search(query_vec, top_k)
    # 根据索引获取对应的文本块
    retrieved_chunks = [doc_chunks[i] for i in indices[0]]
    return retrieved_chunks


def load_BERT_model():
    tokenizer = AutoTokenizer.from_pretrained(FinBERT)
    model = AutoModel.from_pretrained(FinBERT)
    model.to('cpu')
    model.eval()  # 设为评估模式
    return tokenizer, model


if __name__ == '__main__':

    # 加载全局资源
    # embed_model = load_embed_model()
    tokenizer, model = load_BERT_model()
    # index = load_faiss_index(sp_group_name_index)
    index = load_faiss_index('./sp_group_name_v3.index')
    doc_chunks = read_doc_chunks_cache()

    # 示例检索
    query = "中国银行"
    # query = "建设北京分行"
    # query = "胜发证券股份有限公司"
    print(f"start search: {datetime.now()}")
    # retrieved = retrieve(query, bert_model, index, doc_chunks, top_k=5)
    retrieved = retrieve_bert(query, tokenizer, model, index, doc_chunks, top_k=5)
    print(f"end search: {datetime.now()}")
    print(f"type: {type(retrieved)}")
    print("检索到的上下文:")
    for chunk in retrieved:
        print(chunk)
        print("-" * 40)

    # import time
    # import traceback
    #
    # # 加载全局资源
    # log_info.info(f"开始加载全局资源.....")
    # print(f"开始加载全局资源....")
    # embed_model = load_embed_model()
    # index = load_faiss_index(sp_group_name_index)
    # doc_chunks = read_doc_chunks_cache()
    # log_info.info(f"全局资源加载完成....")
    # print(f"全局资源加载完成....")
    #
    # Redis = RedisClient().get_client()
    # while True:
    #     try:
    #         search_inst_list = Redis.lrange('InstTextVector', 0, -1)
    #         Redis.delete('InstTextVector')
    #         if search_inst_list and len(search_inst_list):
    #             for inst_name in search_inst_list:
    #                 retrieved = retrieve(inst_name, embed_model, index, doc_chunks, top_k=5)
    #                 full_name_list = [item.split(',')[0] for item in retrieved if ',' in item]
    #                 full_name_str = ",".join(full_name_list)
    #                 # 写入Redis
    #                 redis_key = f"InstTextVector-{inst_name}"
    #                 Redis.set(redis_key, full_name_str)
    #                 expire_time = 60 * 60 * 10  # 设置10个小时后过期
    #                 Redis.expire(redis_key, expire_time)
    #     except:
    #         traceback.print_exc()
    #         log_error.error(f"{traceback.format_exc()}")
    #     finally:
    #         time.sleep(2)
