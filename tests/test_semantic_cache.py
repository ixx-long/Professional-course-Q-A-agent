"""
单元测试 - semantic_cache 模块
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from src.semantic_cache import SemanticCache


class TestSemanticCache:
    """SemanticCache 单元测试类"""

    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """创建测试用缓存实例"""
        return SemanticCache(
            cache_dir=temp_cache_dir,
            similarity_threshold=0.9,
            max_cache_size=5
        )

    def test_init_creates_directory(self, temp_cache_dir):
        """测试初始化时创建缓存目录"""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        cache = SemanticCache(cache_dir=str(cache_dir))
        assert cache_dir.exists()

    def test_put_and_get_exact_match(self, cache):
        """测试精确匹配（相似度=1.0）"""
        question = "什么是二叉树？"
        embedding = [1.0, 0.0, 0.0]
        answer = "二叉树是一种数据结构"
        sources = [{"file": "test.pdf", "page": 1}]

        cache.put(question, embedding, answer, sources)
        result = cache.get(question, embedding)

        assert result is not None
        assert result["answer"] == answer
        assert result["sources"] == sources
        assert result["cached"] is True
        assert result["similarity"] == 1.0

    def test_get_similar_question(self, cache):
        """测试相似问题匹配"""
        question1 = "什么是二叉树？"
        embedding1 = [1.0, 0.0, 0.0]
        answer1 = "二叉树是一种数据结构"

        cache.put(question1, embedding1, answer1, [])

        # 相似问题（余弦相似度 > 0.9）
        question2 = "二叉树是什么？"
        embedding2 = [0.95, 0.05, 0.0]  # 与 embedding1 相似度高
        result = cache.get(question2, embedding2)

        assert result is not None
        assert result["answer"] == answer1

    def test_get_dissimilar_question(self, cache):
        """测试不相似问题不匹配"""
        question1 = "什么是二叉树？"
        embedding1 = [1.0, 0.0, 0.0]
        answer1 = "二叉树是一种数据结构"

        cache.put(question1, embedding1, answer1, [])

        # 不相似问题（余弦相似度 < 0.9）
        question2 = "什么是操作系统？"
        embedding2 = [0.0, 1.0, 0.0]  # 与 embedding1 正交
        result = cache.get(question2, embedding2)

        assert result is None

    def test_get_empty_cache(self, cache):
        """测试空缓存查询"""
        result = cache.get("任意问题", [1.0, 0.0, 0.0])
        assert result is None

    def test_max_cache_size_eviction(self, cache):
        """测试缓存满时淘汰最旧条目"""
        # 填满缓存（max_cache_size=5）
        for i in range(5):
            cache.put(f"问题{i}", [float(i), 0.0, 0.0], f"答案{i}", [])

        assert len(cache.cache) == 5

        # 添加第6个，应淘汰最旧的
        cache.put("新问题", [5.0, 0.0, 0.0], "新答案", [])
        assert len(cache.cache) == 5

    def test_clear(self, cache):
        """测试清空缓存"""
        cache.put("问题1", [1.0, 0.0, 0.0], "答案1", [])
        cache.put("问题2", [0.0, 1.0, 0.0], "答案2", [])

        cache.clear()
        assert len(cache.cache) == 0

    def test_stats(self, cache):
        """测试获取缓存统计信息"""
        cache.put("问题", [1.0, 0.0, 0.0], "答案", [])

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["similarity_threshold"] == 0.9
        assert "cache_dir" in stats

    def test_persistence(self, temp_cache_dir):
        """测试缓存持久化到磁盘"""
        # 创建缓存并添加数据
        cache1 = SemanticCache(
            cache_dir=temp_cache_dir,
            similarity_threshold=0.9,
            max_cache_size=5
        )
        cache1.put("问题", [1.0, 0.0, 0.0], "答案", [])
        cache1._save_cache()  # 强制保存

        # 创建新实例，应加载已保存的数据
        cache2 = SemanticCache(
            cache_dir=temp_cache_dir,
            similarity_threshold=0.9,
            max_cache_size=5
        )
        result = cache2.get("问题", [1.0, 0.0, 0.0])
        assert result is not None
        assert result["answer"] == "答案"

    def test_cosine_similarity_edge_cases(self, cache):
        """测试余弦相似度边界情况"""
        # 零向量
        similarity = cache._cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert similarity == 0.0

        # 相同向量
        similarity = cache._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert similarity == 1.0

        # 正交向量
        similarity = cache._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert similarity == 0.0

        # 反向向量
        similarity = cache._cosine_similarity([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        assert similarity == -1.0

    def test_compute_hash(self, cache):
        """测试哈希计算"""
        hash1 = cache._compute_hash("问题1")
        hash2 = cache._compute_hash("问题1")
        hash3 = cache._compute_hash("问题2")

        assert hash1 == hash2  # 相同文本哈希相同
        assert hash1 != hash3  # 不同文本哈希不同
        assert len(hash1) == 32  # MD5 哈希长度为 32

    def test_sources_type_consistency(self, cache):
        """测试 sources 类型一致性"""
        # 正常情况：sources 是列表
        cache.put("问题", [1.0, 0.0, 0.0], "答案", [{"file": "test.pdf"}])
        result = cache.get("问题", [1.0, 0.0, 0.0])
        assert isinstance(result["sources"], list)

        # 边界情况：sources 为空列表
        cache.put("问题2", [0.0, 1.0, 0.0], "答案2", [])
        result = cache.get("问题2", [0.0, 1.0, 0.0])
        assert isinstance(result["sources"], list)
        assert result["sources"] == []
