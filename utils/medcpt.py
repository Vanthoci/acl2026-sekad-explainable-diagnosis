import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Dict
import os
import json
from yaspin import yaspin


class MedCPTRetriever:
    def __init__(self, kg_name='direct_exp_m.json', limit=None):
        # 加载 MedCPT 的 Query Encoder 和 Article Encoder
        self.query_model = AutoModel.from_pretrained(
            "ncbi/MedCPT-Query-Encoder")
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            "ncbi/MedCPT-Query-Encoder")
        self.article_model = AutoModel.from_pretrained(
            "ncbi/MedCPT-Article-Encoder")
        self.article_tokenizer = AutoTokenizer.from_pretrained(
            "ncbi/MedCPT-Article-Encoder")

        # 添加设备支持
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_model.to(self.device)
        self.article_model.to(self.device)

        # with yaspin(text="Loading Retriever…"):

        self.index_to_content: Dict[str, str] = {}  # index (str) -> content (str)
        self.index_embeddings: List[np.ndarray] = []  # 存储嵌入向量

        self.memory = []
        now_dir = os.path.dirname(os.path.abspath(__file__))
        # memory_path = os.path.join(now_dir, '..', 'memory', 'memory_gast_merged.json')
        memory_path = os.path.join(now_dir, '..', 'memory', kg_name)
        with open(memory_path, "r") as f:
            self.memory = json.load(f)

        if limit:
            self.memory = self.memory[:limit]

        from tqdm import tqdm
        for m in tqdm(self.memory):
            # self.insert(f"{m['observation']}\n{m['explanation']}", m)
            self.insert(f"{m['explanation']}", m)

        print("MedCPTRetriever initialized.")

    def _encode_text(self, text: str, is_query: bool = True) -> np.ndarray:
        """将文本编码为嵌入向量"""
        # 根据是否是查询选择对应的模型和分词器
        model = self.query_model if is_query else self.article_model
        tokenizer = self.query_tokenizer if is_query else self.article_tokenizer

        # 编码文本
        with torch.no_grad():
            encoded = tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=512  # MedCPT 支持的最大序列长度
            ).to(self.device)  # 添加设备指定
            # 获取 [CLS] token 的嵌入 (last hidden state)
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            return embeds.squeeze().cpu().numpy()

    def insert(self, index: str, content: object):
        """插入 (index, content) 对，其中 index 用于语义相似性比较"""
        if index in self.index_to_content:
            return False
            # raise ValueError(f"Index '{index}' already exists!")

        # 存储 content
        self.index_to_content[index] = content

        # 计算 index 的嵌入并存储
        embedding = self._encode_text(
            index, is_query=False)  # 使用 Article Encoder 编码 index
        self.index_embeddings.append(embedding)
        return True


    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 except_kv: Tuple[str, str] = None) -> List[Tuple[object, float]]:
        """根据查询返回 top-k 个最匹配的结果，可选地排除特定 key-value 对"""
        if not self.index_embeddings:
            return []

        # 编码查询
        query_embedding = self._encode_text(
            query, is_query=True)  # 使用 Query Encoder 编码查询

        # 计算查询与所有索引嵌入的余弦相似度
        embeddings_array = np.array(self.index_embeddings)
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) *
            np.linalg.norm(query_embedding))

        # 获取 top-k 结果并应用过滤
        top_k_indices = np.argsort(similarities)[::-1]
        results = []
        
        for idx in top_k_indices:
            index = list(self.index_to_content.keys())[idx]
            content = self.index_to_content[index]
            score = float(similarities[idx])
            
            # 检查是否满足排除条件
            if except_kv is not None:
                key, value = except_kv
                if key in content:
                    if content[key].lower() == value.lower():
                        continue
                    
            results.append((content, score))
            if len(results) >= top_k:
                break

        return results

    def group_retrieve(self,
                  queries: List[str],
                  topic: str = "",
                  top_k: int = 5,
                  except_kv: Tuple[str, str] = None) -> List[Tuple[object, float]]:
        """
        Retrieve results for multiple queries, optionally combined with a topic.
        Results from all queries are combined and sorted by score to return top_k results.
        
        Args:
            queries: List of query strings to search for
            topic: Optional topic string to append to each query
            top_k: Number of top results to return overall
            except_kv: Optional tuple of (key, value) to exclude from results
        
        Returns:
            List of (content, score) tuples sorted by score in descending order
        """
        all_results = []

        for query in queries:
            full_query = f"{query} {topic}".strip() if topic else query
            query_results = self.retrieve(full_query, top_k=top_k, except_kv=except_kv)
            all_results.extend(query_results)

        # Sort all results by score in descending order
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in all_results:
            # Using the content as the unique identifier
            content_id = str(result[0])  # Convert content to string for hashing
            if content_id not in seen:
                seen.add(content_id)
                unique_results.append(result)

        # Return top_k unique results
        return unique_results[:top_k]



