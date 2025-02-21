#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@CreateTime    : 2025/2/21 10:26
@Author  : wang yi ming
@file for: 
"""

import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import faiss
import numpy as np
from tqdm import tqdm

from config.local_config import FinBERT
from utils.common_util import cache_doc_chunks, get_text_chunks, get_data_documents


def get_embedding(text, tokenizer, model, device, max_length=128):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # 确保输入在 GPU 上
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用pooler_output作为句子向量
    # embedding = outputs.pooler_output[0].numpy()
    embedding = outputs.pooler_output[0].cpu().numpy()  # 从 GPU 拷回 CPU
    return embedding


# def text_vectorization(doc_chunks, tokenizer, model):
#     try:
#         batch_size = 64
#         embeddings = []
#         with tqdm(total=len(doc_chunks)) as pbar:
#             for i in range(0, len(doc_chunks), batch_size):
#                 corpus = doc_chunks[i:i + batch_size]
#                 embeddings.extend([get_embedding(doc, tokenizer, model) for doc in corpus])
#                 pbar.update(batch_size)
#     except Exception as e:
#         print(f"文本向量化失败: {e}")
#         raise e
#     print(f"生成了 {len(embeddings)} 个向量")
#     return embeddings


def text_vectorization(doc_chunks, tokenizer, model, device, batch_size=64):
    embeddings = []
    with tqdm(total=len(doc_chunks)) as pbar:
        for i in range(0, len(doc_chunks), batch_size):
            batch_texts = doc_chunks[i:i + batch_size]

            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}  # 确保输入在 GPU 上

            with torch.no_grad():
                outputs = model(**inputs)

            batch_embeddings = outputs.pooler_output.cpu().numpy()  # 确保最终数据返回 CPU
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch_texts))

    print(f"生成了 {len(embeddings)} 个向量")
    return embeddings


# # 3. 利用FAISS构建向量索引
# d = corpus_embeddings.shape[1]  # 向量维度
# index = faiss.IndexFlatL2(d)  # 使用L2距离
# index.add(corpus_embeddings)  # 将文档向量添加到索引中
#
# # 4. 对用户查询进行编码并检索最相似文档
# query = "怎样用BERT和FAISS做问答系统？"
# query_embedding = np.array([get_embedding(query)]).astype('float32')
# k = 3  # 检索top-3个最相似文档
# distances, indices = index.search(query_embedding, k)
#
# retrieved_docs = [corpus[i] for i in indices[0]]
# print("检索到的文档：", retrieved_docs)


def save_to_vector_database(embeddings):
    try:
        vectors = np.array(embeddings).astype('float32')
        d = vectors.shape[1]

        index = faiss.IndexFlatL2(d)
        index.add(vectors)
        print(f"向量库中现有向量数: {index.ntotal}")
        faiss.write_index(index, "sp_group_name_v3.index")
    except Exception as e:
        print(f"保存向量索引失败: {e}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载BERT模型和分词器
    # tokenizer = BertTokenizer.from_pretrained(FinBERT)
    # # model = BertModel.from_pretrained(FinBERT)
    # model = BertModel.from_pretrained(FinBERT, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(FinBERT)
    model = AutoModel.from_pretrained(FinBERT)
    model.to(device)
    model.eval()  # 设为评估模式

    documents = get_data_documents()

    doc_chunks = get_text_chunks(documents)
    embeddings = text_vectorization(doc_chunks, tokenizer, model, device)
    save_to_vector_database(embeddings)
    cache_doc_chunks(doc_chunks)


if __name__ == '__main__':
    main()
