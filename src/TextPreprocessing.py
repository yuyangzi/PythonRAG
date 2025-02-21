#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@CreateTime    : 2025/2/12 09:55
@Author        : wang yi ming
@file for      : 构建文本块、向量化、构建 FAISS 索引并缓存结果
"""

import os
import pickle
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
from tqdm import tqdm

from config.local_config import text2vec_bge_large_chinese


def load_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    doc_chunks = []
    for doc in documents:
        try:
            chunks = text_splitter.split_text(doc)
            doc_chunks.extend(chunks)
        except Exception as e:
            print(f"分割文档时出错: {e}")
    print(f"共获得 {len(doc_chunks)} 个文本块")
    return doc_chunks


def text_vectorization(doc_chunks):
    try:
        # 加载嵌入模型
        # embed_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        embed_model = SentenceTransformer(text2vec_bge_large_chinese)
        # # 如果有多个 GPU，使用 DataParallel 将模型并行化
        # if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        #     print(f"使用 {torch.cuda.device_count()} 块 GPU")
        #     embed_model = torch.nn.DataParallel(embed_model)  # 自动使用所有可用的 GPU
        # 将模型转移到 cuda
        embed_model = embed_model.to('cuda')
        # 半精度
        # embed_model = embed_model.half()
    except Exception as e:
        print(f"加载嵌入模型失败: {e}")
        raise e
    # 对所有文本块生成向量
    try:
        batch_size = 64
        embeddings = []
        with tqdm(total=len(doc_chunks)) as pbar:
            for i in range(0, len(doc_chunks), batch_size):
                batch = doc_chunks[i:i + batch_size]
                # embeddings.extend(embed_model.module.encode(batch, convert_to_tensor=False))
                embeddings.extend(embed_model.encode(batch, convert_to_tensor=False))
                pbar.update(batch_size)
    except Exception as e:
        print(f"文本向量化失败: {e}")
        raise e
    print(f"生成了 {len(embeddings)} 个向量")
    return embeddings


def save_to_vector_database(embeddings, nlist=100, nprobe=10):
    try:
        vectors = np.array(embeddings).astype('float32')
        d = vectors.shape[1]

        index = faiss.IndexFlatL2(d)
        index.add(vectors)
        print(f"向量库中现有向量数: {index.ntotal}")
        faiss.write_index(index, "sp_group_name_v2.index")

        # # 创建倒排文件索引（IVF）
        # quantizer = faiss.IndexFlatL2(d)  # 基础索引，使用 L2 距离
        # index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        #
        # # 训练索引（使用一些样本向量进行训练）
        # print("开始训练索引...")
        # index_ivf.train(vectors)  # 使用所有向量训练索引，或者使用样本数据进行训练
        # print("索引训练完成")
        #
        # # 添加向量到索引
        # index_ivf.add(vectors)  # 添加向量到索引
        # print(f"向量库中现有向量数: {index_ivf.ntotal}")
        #
        # # 设置查询时使用的簇数量
        # index_ivf.nprobe = nprobe
        # print(f"查询时将搜索 {nprobe} 个簇")
        #
        # # 将训练好的索引保存到磁盘
        # faiss.write_index(index_ivf, "local_knowledge_index_ivf.index")
        # print("索引已保存")

    except Exception as e:
        print(f"保存向量索引失败: {e}")


def cache_doc_chunks(doc_chunks):
    cache_dir = os.path.join('..', 'pickle_cached')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'doc_chunks_cache.pkl')
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(doc_chunks, f)
        print(f"成功缓存文本块到 {cache_path}")
    except Exception as e:
        print(f"缓存文本块失败: {e}")


if __name__ == '__main__':
    data_dir = "../data"
    documents = []
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                text = load_text_file(os.path.join(data_dir, filename))
                text_lit = text.split('\n')
                print(f"text_lit: {len(text_lit)}")
                for text_item in text_lit:
                    documents.append(text_item)
    except Exception as e:
        print(f"读取数据目录失败: {e}")

    doc_chunks = get_text_chunks(documents)
    embeddings = text_vectorization(doc_chunks)
    save_to_vector_database(embeddings)
    cache_doc_chunks(doc_chunks)
