from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import jieba
import json
from transformers import AutoTokenizer, GenerationConfig
from transformers import AutoModelForSequenceClassification
import torch

import time

import wordsSimilar


def remove_duplicate(lst):
    dup = {}
    for l in lst:
        dup[l.replace(" ", "")] = l
    return dup.values()


def hfLoading():
    model_name = rf"/root/zhanxin/code/recommend/Xorbits_new/bge-large-zh-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    query_instruction=""
    print("start")
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction=query_instruction
    )
    print("finish")
    return hf


def build_index(hf, docs):
    # 生成文档向量
    bm25_retriever = BM25Retriever.from_texts([" ".join(jieba.lcut(q)) for q in docs])
    vectordb = FAISS.from_texts(texts=docs, embedding=hf)
    print("index built")
    ranktokenizer = AutoTokenizer.from_pretrained(rf'/root/zhanxin/code/recommend/Xorbits_new/bge-reranker-base')
    rankmodel = AutoModelForSequenceClassification.from_pretrained(rf'/root/zhanxin/code/recommend/Xorbits_new/bge-reranker-base')
    print("finish")
    rankmodel.eval()
    return bm25_retriever, vectordb, ranktokenizer, rankmodel


def search_index(bm25_retriever, vectordb, ranktokenizer, rankmodel, query):
    start = time.time()
    print("开始检索")
    bm25_result = bm25_retriever.get_relevant_documents(" ".join(jieba.lcut(query)))
    bm25_result = [d.page_content.replace(" ", "") for d in bm25_result]
    # print("bm25_result", bm25_result)
    vector_result = vectordb.similarity_search(query, k=5)
    vector_result = [d.page_content for d in vector_result]
    # print("vector_result", vector_result)
    results = remove_duplicate(set(vector_result + bm25_result))
    pairs = [[query, q] for q in results]
    end = time.time()
    print("检索结束，用时： ", end - start)
    
    with torch.no_grad():
        inputs = ranktokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = rankmodel(**inputs, return_dict=True).logits.view(-1, ).float().numpy().tolist()
        results = [(q, s) for q, s in zip(results, scores)]
        results = sorted(results, key=lambda x: x[1], reverse=True)

        return results[0][0] if results != None else ""
        # print(results)
    #     for r in results:
    #         if r[1] > QUERY_THRESHOLD:
    #             new_col.append(r[0])
    # return new_col

