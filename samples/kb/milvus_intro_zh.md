# Milvus 简介

Milvus 是一个开源的向量数据库，由 Zilliz 发起，现为 LF AI & Data 基金会项目。它专为海量向量的相似度检索设计，可以存储、索引和检索由深度学习模型生成的 embedding 向量。

## 部署形态

Milvus 提供三种部署形态：Milvus Lite（嵌入式，适合原型验证，pip 安装即可）、Milvus Standalone（单机 Docker 部署）和 Milvus Distributed（基于 Kubernetes 的分布式集群）。托管服务则是 Zilliz Cloud。

## 索引类型

Milvus 支持多种向量索引：

- FLAT：暴力检索，召回率 100%，适合小数据量。
- IVF_FLAT / IVF_SQ8 / IVF_PQ：基于倒排聚类，nlist 控制聚类数量，nprobe 控制查询时扫描的聚类数。
- HNSW：基于图的索引，参数 M 控制每个节点的邻居数（常用 16），efConstruction 控制建图质量（常用 200），查询参数 ef 越大召回越高。
- DISKANN：基于磁盘的图索引，适合内存放不下的大规模数据。
- AUTOINDEX：由 Milvus 自动选择索引和参数，推荐新手使用。

稀疏向量使用 SPARSE_INVERTED_INDEX，配合 BM25 度量实现全文检索。

## 度量类型

稠密向量常用 COSINE（余弦相似度）、IP（内积）和 L2（欧氏距离）。文本 embedding 一般推荐 COSINE。

## 一致性级别

Milvus 支持 Strong、Bounded、Session 和 Eventually 四种一致性级别，默认是 Bounded。需要写入后立即可查时使用 Strong。
