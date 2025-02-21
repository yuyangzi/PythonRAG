#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@CreateTime    : 2025/2/21 13:25
@Author  : wang yi ming
@file for: 
"""
import os
import pickle

from langchain_text_splitters import CharacterTextSplitter

from config.local_config import data_dir


def load_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


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


def get_data_documents():
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
    finally:
        return []
