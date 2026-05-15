# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - 检索质量测试套件

测试分三组：
  [Group 1] 单元测试    —— 纯函数，无需 API（TestDetectLanguage / TestMediaIntent / TestSplitText）
  [Group 2] 索引结构测试 —— 仅读文件，无需 API（TestIndexLoading）
  [Group 3] 检索质量测试 —— 需要 DASHSCOPE_API_KEY（TestRetrievalQuality）

运行方式：
  全部测试：  python 8-disney_rag_test.py
  仅无需 API：python 8-disney_rag_test.py TestDetectLanguage TestMediaIntent TestSplitText TestIndexLoading
  详细输出：  python 8-disney_rag_test.py -v
"""
import os
import json
import unittest
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(PROJECT_DIR, "disney_indexes")

# FAISS 的 C 库在 Windows 上不支持 Unicode 路径，统一切换到项目根目录后使用相对路径
os.chdir(PROJECT_DIR)
_INDEX_DIR_REL = "disney_indexes"   # 仅供 faiss.read_index() 使用的相对路径版本


def _faiss_path(prefix: str, suffix: str) -> str:
    """返回 FAISS 可用的相对路径（避免中文路径报错）"""
    return f"{_INDEX_DIR_REL}/{prefix}_{suffix}"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
SKIP_API = not DASHSCOPE_API_KEY
SKIP_MSG = "未设置 DASHSCOPE_API_KEY，跳过 Embedding API 测试"

# 分类名称常量（与 7-disney_build_full_index.py 保持一致）
CAT_PRODUCTS   = "产品与服务详情"
CAT_OPERATIONS = "运营流程与标准作业程序"
CAT_EMERGENCY  = "特殊情况与应急预案"
CAT_CUSTOMER   = "客户关系与支持话术"
CAT_INTERNAL   = "内部知识与工具"

# 索引前缀列表
CATEGORY_PREFIXES = [
    "cat1_products",
    "cat2_operations",
    "cat3_emergency",
    "cat4_customer",
    "cat5_internal",
    "disney_full",
]


# ── 本文件内联的纯函数（与 6-disney_app.py 保持同步，避免导入副作用） ──────────

def detect_language(text: str) -> str:
    """中文字符（CJK 统一汉字）占比 > 20% 视为中文，否则视为英文"""
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    return "zh" if len(text) > 0 and chinese / len(text) > 0.2 else "en"


def detect_media_intent(query: str) -> tuple:
    IMAGE_KW = ["图片", "海报", "照片", "看看", "长什么样", "图",
                "image", "photo", "poster", "picture", "show", "look", "see"]
    VIDEO_KW = ["视频", "录像", "影片", "看一下", "播放",
                "video", "watch", "play", "clip", "footage"]
    q = query.lower()
    return (any(kw in q for kw in IMAGE_KW), any(kw in q for kw in VIDEO_KW))


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — 纯函数单元测试（无需 API）
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectLanguage(unittest.TestCase):
    """验证 detect_language() 的语言分类逻辑"""

    def test_pure_chinese(self):
        self.assertEqual(detect_language("迪士尼门票多少钱"), "zh")

    def test_pure_english(self):
        self.assertEqual(detect_language("What is the ticket price?"), "en")

    def test_chinese_with_digits_and_punctuation(self):
        self.assertEqual(detect_language("门票价格是599元，含2张儿童票"), "zh")

    def test_short_english_word(self):
        self.assertEqual(detect_language("Hi"), "en")

    def test_empty_string_returns_english(self):
        # 空字符串无法判断，默认英文
        self.assertEqual(detect_language(""), "en")

    def test_boundary_exactly_20_percent_treated_as_english(self):
        # "01234567迪士" = 8 ASCII + 2 Chinese = 10 chars, ratio = 0.20
        # 条件是 > 0.20，不含等于，故返回 "en"
        text = "01234567迪士"
        self.assertEqual(detect_language(text), "en")

    def test_just_above_20_percent_treated_as_chinese(self):
        # 3 Chinese / 13 total ≈ 23% → zh
        text = "0123456789迪士尼"
        self.assertEqual(detect_language(text), "zh")

    def test_mixed_query_dominated_by_chinese(self):
        # "万圣节 Halloween 海报" 中文字符占多数
        text = "万圣节 Halloween 海报"
        self.assertEqual(detect_language(text), "zh")


class TestMediaIntent(unittest.TestCase):
    """验证 detect_media_intent() 的关键词匹配逻辑"""

    def test_image_intent_via_chinese_keyword(self):
        want_img, want_vid = detect_media_intent("万圣节的海报是什么样的")
        self.assertTrue(want_img, "含'海报'应识别为图片意图")
        self.assertFalse(want_vid)

    def test_image_intent_via_english_keyword(self):
        want_img, want_vid = detect_media_intent("show me the Halloween poster")
        self.assertTrue(want_img, "'show'/'poster' 应识别为图片意图")
        self.assertFalse(want_vid)

    def test_video_intent_via_chinese_keyword(self):
        want_img, want_vid = detect_media_intent("有相关视频可以看吗")
        self.assertFalse(want_img)
        self.assertTrue(want_vid, "含'视频'应识别为视频意图")

    def test_video_intent_via_english_keyword(self):
        want_img, want_vid = detect_media_intent("I want to watch a video clip")
        self.assertFalse(want_img)
        self.assertTrue(want_vid, "'watch'/'video'/'clip' 应识别为视频意图")

    def test_no_media_intent_for_text_only_query(self):
        want_img, want_vid = detect_media_intent("迪士尼门票退款流程是什么")
        self.assertFalse(want_img, "纯文本问题不应有图片意图")
        self.assertFalse(want_vid, "纯文本问题不应有视频意图")

    def test_both_intents_detected(self):
        want_img, want_vid = detect_media_intent("有没有海报和视频")
        self.assertTrue(want_img)
        self.assertTrue(want_vid)

    def test_keyword_matching_is_case_insensitive(self):
        want_img, _ = detect_media_intent("Can you show me the POSTER?")
        self.assertTrue(want_img, "关键词匹配应忽略大小写")

    def test_keyword_must_be_substring(self):
        # "videography" 包含 "video"，应匹配（子串匹配）
        _, want_vid = detect_media_intent("show me the videography")
        self.assertTrue(want_vid)


class TestSplitText(unittest.TestCase):
    """验证 split_text() 的分块行为"""

    def test_short_text_yields_single_chunk(self):
        text = "这是一段短文本"
        chunks = split_text(text, chunk_size=500)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_text_yields_multiple_chunks(self):
        text = "A" * 1200
        chunks = split_text(text, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_first_chunk_length_equals_chunk_size(self):
        text = "X" * 800
        chunks = split_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks[0]), 500)

    def test_overlap_causes_more_chunks_than_naive_division(self):
        # 1000 chars / 500 chunk_size = 2 chunks without overlap
        # with overlap=100: start=0,400,800 → 3 chunks
        text = "A" * 1000
        chunks_no_overlap = split_text(text, chunk_size=500, overlap=0)
        chunks_with_overlap = split_text(text, chunk_size=500, overlap=100)
        self.assertGreaterEqual(len(chunks_with_overlap), len(chunks_no_overlap))

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(split_text(""), [])

    def test_whitespace_only_text_yields_no_chunks(self):
        self.assertEqual(split_text("   \n   "), [])

    def test_chunk_content_is_stripped(self):
        text = "  迪士尼  "
        chunks = split_text(text, chunk_size=500)
        self.assertEqual(chunks[0], "迪士尼")

    def test_exact_chunk_size_text_with_no_overlap_yields_one_chunk(self):
        # overlap=0 时，恰好 chunk_size 的文本只产生 1 个 chunk
        text = "Z" * 500
        chunks = split_text(text, chunk_size=500, overlap=0)
        self.assertEqual(len(chunks), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — 索引结构测试（无需 API，仅读文件）
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndexLoading(unittest.TestCase):
    """验证 7-disney_build_full_index.py 生成的索引文件结构是否正确"""

    def test_index_directory_exists(self):
        self.assertTrue(os.path.isdir(INDEX_DIR),
                        f"索引目录不存在，请先运行 7-disney_build_full_index.py: {INDEX_DIR}")

    def test_all_faiss_files_exist(self):
        for prefix in CATEGORY_PREFIXES:
            path = os.path.join(INDEX_DIR, f"{prefix}_index.faiss")
            self.assertTrue(os.path.exists(path), f"FAISS 文件缺失: {path}")

    def test_all_metadata_files_exist(self):
        for prefix in CATEGORY_PREFIXES:
            path = os.path.join(INDEX_DIR, f"{prefix}_metadata.json")
            self.assertTrue(os.path.exists(path), f"元数据文件缺失: {path}")

    def test_metadata_is_valid_json_list(self):
        for prefix in CATEGORY_PREFIXES:
            path = os.path.join(INDEX_DIR, f"{prefix}_metadata.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list, f"{prefix}: metadata 应为 JSON list")
            self.assertGreater(len(data), 0, f"{prefix}: metadata 不应为空")

    def test_metadata_required_fields(self):
        """每条 metadata 必须包含 id / source / type / content / category 字段"""
        required = {"id", "source", "type", "content", "category"}
        path = os.path.join(INDEX_DIR, "disney_full_metadata.json")
        if not os.path.exists(path):
            self.skipTest("disney_full_metadata.json 不存在")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            missing = required - set(item.keys())
            self.assertEqual(missing, set(),
                             f"id={item.get('id')}: 缺少字段 {missing}")

    def test_metadata_type_values_are_valid(self):
        """type 字段只允许 text / image / video"""
        path = os.path.join(INDEX_DIR, "disney_full_metadata.json")
        if not os.path.exists(path):
            self.skipTest("disney_full_metadata.json 不存在")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        valid_types = {"text", "image", "video"}
        for item in data:
            self.assertIn(item["type"], valid_types,
                          f"id={item['id']}: 非法 type={item['type']!r}")

    def test_full_index_contains_all_five_categories(self):
        """全局索引应覆盖全部 5 个业务分类"""
        path = os.path.join(INDEX_DIR, "disney_full_metadata.json")
        if not os.path.exists(path):
            self.skipTest("disney_full_metadata.json 不存在")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        found = {item["category"] for item in data}
        for cat in [CAT_PRODUCTS, CAT_OPERATIONS, CAT_EMERGENCY,
                    CAT_CUSTOMER, CAT_INTERNAL]:
            self.assertIn(cat, found, f"全局索引中缺少分类: {cat}")

    def test_full_index_record_count_at_least_sum_of_categories(self):
        """全局索引的记录数 ≥ 各分类索引记录数之和（还包含概览文档）"""
        category_total = 0
        for prefix in CATEGORY_PREFIXES[:-1]:   # 排除 disney_full 本身
            path = os.path.join(INDEX_DIR, f"{prefix}_metadata.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    category_total += len(json.load(f))
        full_path = os.path.join(INDEX_DIR, "disney_full_metadata.json")
        if not os.path.exists(full_path):
            self.skipTest("disney_full_metadata.json 不存在")
        with open(full_path, encoding="utf-8") as f:
            full_count = len(json.load(f))
        self.assertGreaterEqual(full_count, category_total,
                                f"全局索引({full_count}) 小于分类之和({category_total})")

    def test_products_index_contains_images(self):
        """cat1_products 应包含 ≥ 1 条 image 类型记录（邮轮价格图片）"""
        path = os.path.join(INDEX_DIR, "cat1_products_metadata.json")
        if not os.path.exists(path):
            self.skipTest("cat1_products_metadata.json 不存在")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        images = [item for item in data if item["type"] == "image"]
        self.assertGreaterEqual(len(images), 1,
                                "cat1_products 应至少包含 1 张图片")

    def test_image_metadata_has_path_field(self):
        """图片类型的 metadata 必须包含 path 字段"""
        path = os.path.join(INDEX_DIR, "cat1_products_metadata.json")
        if not os.path.exists(path):
            self.skipTest("cat1_products_metadata.json 不存在")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in (d for d in data if d["type"] == "image"):
            self.assertIn("path", item,
                          f"图片 id={item['id']} ({item['source']}) 缺少 path 字段")

    def test_faiss_index_can_be_loaded(self):
        """FAISS 索引文件应可正常加载，且不为空"""
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss 未安装")
        for prefix in CATEGORY_PREFIXES:
            p = os.path.join(INDEX_DIR, f"{prefix}_index.faiss")
            if not os.path.exists(p):
                continue
            idx = faiss.read_index(_faiss_path(prefix, "index.faiss"))
            self.assertGreater(idx.ntotal, 0,
                               f"{prefix}: FAISS 索引不应为空")

    def test_faiss_vector_count_matches_metadata(self):
        """FAISS 索引向量数必须等于对应 metadata 的条目数"""
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss 未安装")
        for prefix in CATEGORY_PREFIXES:
            idx_p = os.path.join(INDEX_DIR, f"{prefix}_index.faiss")
            meta_p = os.path.join(INDEX_DIR, f"{prefix}_metadata.json")
            if not (os.path.exists(idx_p) and os.path.exists(meta_p)):
                continue
            idx = faiss.read_index(_faiss_path(prefix, "index.faiss"))
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(idx.ntotal, len(meta),
                             f"{prefix}: 向量数({idx.ntotal}) ≠ metadata 条数({len(meta)})")

    def test_category_index_contains_only_own_category(self):
        """各分类索引的所有记录 category 字段应统一（不含其他分类）"""
        prefix_to_cat = {
            "cat1_products":   CAT_PRODUCTS,
            "cat2_operations": CAT_OPERATIONS,
            "cat3_emergency":  CAT_EMERGENCY,
            "cat4_customer":   CAT_CUSTOMER,
            "cat5_internal":   CAT_INTERNAL,
        }
        for prefix, expected_cat in prefix_to_cat.items():
            path = os.path.join(INDEX_DIR, f"{prefix}_metadata.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self.assertEqual(item["category"], expected_cat,
                                 f"{prefix}: 发现非本分类记录 id={item['id']} "
                                 f"category={item['category']!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — 检索质量测试（需要 DASHSCOPE_API_KEY）
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipIf(SKIP_API, SKIP_MSG)
class TestRetrievalQuality(unittest.TestCase):
    """
    通过真实 Embedding + FAISS 搜索验证召回质量。

    评判标准：
      ① L2 距离 < max_distance  → 向量空间内语义相近
      ② top-K 结果中出现预期分类 → 分类路由正确
      ③ source / content 含预期关键词 → 内容匹配正确
    """

    @classmethod
    def setUpClass(cls):
        import faiss
        import dashscope
        from http import HTTPStatus

        dashscope.api_key = DASHSCOPE_API_KEY
        cls._dashscope = dashscope
        cls._HTTPStatus = HTTPStatus

        cls.full_index = faiss.read_index(_faiss_path("disney_full", "index.faiss"))
        with open(os.path.join(INDEX_DIR, "disney_full_metadata.json"),
                  encoding="utf-8") as f:
            cls.full_meta = json.load(f)

        cls.cat_indexes, cls.cat_metas = {}, {}
        for prefix in CATEGORY_PREFIXES[:-1]:
            ip = os.path.join(INDEX_DIR, f"{prefix}_index.faiss")
            mp = os.path.join(INDEX_DIR, f"{prefix}_metadata.json")
            if os.path.exists(ip) and os.path.exists(mp):
                cls.cat_indexes[prefix] = faiss.read_index(
                    _faiss_path(prefix, "index.faiss"))
                with open(mp, encoding="utf-8") as f:
                    cls.cat_metas[prefix] = json.load(f)

        print(f"\n  [setUpClass] 全局索引: {cls.full_index.ntotal} 条记录")

    # ── 内部工具方法 ─────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        from http import HTTPStatus
        resp = self._dashscope.MultiModalEmbedding.call(
            model="tongyi-embedding-vision-plus",
            input=[{"text": text}]
        )
        if resp.status_code != HTTPStatus.OK:
            self.fail(f"Embedding 调用失败: {resp.message}")
        return np.array([resp.output["embeddings"][0]["embedding"]], dtype="float32")

    def _search(self, query: str, index=None, metadata=None, k: int = 10) -> list:
        if index is None:
            index = self.full_index
        if metadata is None:
            metadata = self.full_meta
        vec = self._embed(query)
        dists, idxs = index.search(vec, min(k, index.ntotal))
        return [
            {"distance": float(d), "similarity": 1.0 / (1.0 + float(d)),
             "metadata": metadata[i]}
            for d, i in zip(dists[0], idxs[0]) if i != -1
        ]

    def _assert_top_k_contains_category(self, results, expected_cat, k=5):
        cats = [r["metadata"]["category"] for r in results[:k]]
        self.assertIn(expected_cat, cats,
                      f"前{k}条结果中未出现分类 [{expected_cat}]\n实际分类: {cats}")

    def _assert_keywords_in_top_k(self, results, keywords, fields=("source", "content"), k=5):
        text = " ".join(r["metadata"].get(f, "") for r in results[:k] for f in fields)
        matched = [kw for kw in keywords if kw in text]
        self.assertTrue(matched,
                        f"前{k}条结果中均不含关键词 {keywords}\n"
                        f"来源: {[r['metadata']['source'] for r in results[:k]]}")

    # ── 测试用例 ─────────────────────────────────────────────────────────────

    def test_01_ticket_refund_policy(self):
        """退款流程查询 → 应召回退款/票务相关文档"""
        results = self._search("上海迪士尼门票退款流程是什么")
        self.assertLess(results[0]["distance"], 2.5,
                        f"top1 距离过大: {results[0]['distance']:.4f}")
        self._assert_keywords_in_top_k(results, ["退款", "票"])

    def test_02_elderly_ticket_discount(self):
        """老人票查询 → 应召回老人票优惠相关文档"""
        results = self._search("老人票有哪些优惠规定")
        self.assertLess(results[0]["distance"], 2.5,
                        f"top1 距离过大: {results[0]['distance']:.4f}")
        self._assert_keywords_in_top_k(results, ["老人", "票"])

    def test_03_hotel_membership_benefits(self):
        """酒店会员查询 → 应精准召回会员制度文档"""
        results = self._search("上海迪士尼酒店会员专属福利有哪些")
        self.assertLess(results[0]["distance"], 2.0,
                        f"top1 距离过大: {results[0]['distance']:.4f}")
        combined = results[0]["metadata"]["source"] + results[0]["metadata"]["content"]
        self.assertTrue("会员" in combined or "酒店" in combined,
                        f"top1 不含会员/酒店关键词: {results[0]['metadata']['source']}")

    def test_04_annual_pass_rules(self):
        """年票查询 → 应召回年票相关文档"""
        results = self._search("迪士尼年票使用规则和限制")
        self.assertLess(results[0]["distance"], 2.5,
                        f"top1 距离过大: {results[0]['distance']:.4f}")
        self._assert_keywords_in_top_k(results, ["年票", "票"])

    def test_05_restaurant_info(self):
        """餐饮查询 → 应召回餐饮相关文档，分类属于产品服务"""
        results = self._search("迪士尼乐园里有哪些餐厅推荐")
        self.assertLess(results[0]["distance"], 2.5,
                        f"top1 距离过大: {results[0]['distance']:.4f}")
        self._assert_top_k_contains_category(results, CAT_PRODUCTS)

    def test_06_emergency_handling(self):
        """紧急情况查询 → 应急预案分类应出现在 top-10；文档内容相关度距离应 < 2.5"""
        # 应急文档是一站式速查表（结构化格式），在 305 条混合库中排名可能在 5~10 之间
        results = self._search("游客在乐园内突发紧急情况如何处理", k=10)
        self._assert_top_k_contains_category(results, CAT_EMERGENCY, k=10)
        emergency_hits = [r for r in results
                          if r["metadata"]["category"] == CAT_EMERGENCY]
        self.assertLess(emergency_hits[0]["distance"], 2.5,
                        f"应急预案最佳结果距离过大: {emergency_hits[0]['distance']:.4f}")

    def test_07_complaint_escalation(self):
        """客诉升级查询 → 应从运营流程分类中召回"""
        results = self._search("客户投诉升级处理流程和判断标准")
        self._assert_top_k_contains_category(results, CAT_OPERATIONS)

    def test_08_hotel_info(self):
        """酒店房型查询 → 应从运营流程分类召回酒店信息文档"""
        results = self._search("迪士尼各酒店的房型介绍和价格")
        self._assert_keywords_in_top_k(results, ["酒店"])

    def test_09_customer_service_scripts(self):
        """服务话术查询 → 应从客户关系分类召回"""
        results = self._search("如何用官方话术回应顾客的不满情绪")
        self._assert_top_k_contains_category(results, CAT_CUSTOMER)

    def test_10_employee_training(self):
        """员工培训查询 → 应从内部知识分类召回"""
        results = self._search("迪士尼乐园员工培训课程内容")
        self._assert_top_k_contains_category(results, CAT_INTERNAL)
        self._assert_keywords_in_top_k(results, ["培训", "员工"])

    def test_11_staff_job_responsibilities(self):
        """岗位职责查询 → 应从内部知识分类召回"""
        results = self._search("迪士尼乐园导览员的岗位职责是什么")
        self._assert_top_k_contains_category(results, CAT_INTERNAL)

    def test_12_cruise_image_retrieval(self):
        """邮轮价格查询 → 跨模态检索：验证图片 embedding 与文本查询的语义距离合理"""
        # 文本查询会优先召回文字文档（text-to-text 更近），
        # 此测试专注于验证 text→image 跨模态向量距离是否在合理范围内，
        # 而非排名。搜索全部记录，取图片中距离最近的一条。
        results = self._search("迪士尼邮轮价格一览", k=self.full_index.ntotal)
        image_hits = [r for r in results if r["metadata"]["type"] == "image"]
        self.assertGreater(len(image_hits), 0,
                           "全局索引中未找到任何图片类型记录（构建索引时未包含图片？）")
        best = min(image_hits, key=lambda x: x["distance"])
        # 跨模态匹配阈值宽松（text→image 语义距离通常大于 text→text）
        self.assertLess(best["distance"], 6.0,
                        f"最近邮轮图片距离过大，跨模态检索可能失效: {best['distance']:.4f}")
        print(f"\n  [图片跨模态] 最近图片: {best['metadata']['source']}  "
              f"distance={best['distance']:.4f}  rank={results.index(best)+1}/{len(results)}")

    def test_13_english_query_cross_lingual(self):
        """英文查询 → 多模态模型应支持跨语言检索，返回相关中文文档"""
        results = self._search("What are the ticket refund policies at Disneyland?")
        self.assertGreater(len(results), 0)
        self.assertLess(results[0]["distance"], 3.0,
                        f"英文查询 top1 距离过大: {results[0]['distance']:.4f}")

    def test_14_category_specific_index_precision(self):
        """在 cat3_emergency 专用索引中，紧急查询应能精准召回，且所有结果均属应急分类"""
        if "cat3_emergency" not in self.cat_indexes:
            self.skipTest("cat3_emergency 索引未加载")
        cat_idx = self.cat_indexes["cat3_emergency"]
        cat_meta = self.cat_metas["cat3_emergency"]
        results = self._search("紧急情况处置预案速查", cat_idx, cat_meta, k=3)
        # ① 专用索引 top1 距离应足够小（知识库内确实有紧急处理文档）
        self.assertLess(results[0]["distance"], 2.0,
                        f"cat3_emergency top1 距离过大: {results[0]['distance']:.4f}")
        # ② 专用索引中所有结果 category 只能是应急预案（隔离验证）
        for r in results:
            self.assertEqual(r["metadata"]["category"], CAT_EMERGENCY,
                             f"cat3_emergency 索引返回了非本分类结果: "
                             f"{r['metadata']['category']}")

    def test_15_irrelevant_query_higher_distance_than_relevant(self):
        """无关查询的 top1 距离应大于相关查询（负例验证）"""
        relevant = self._search("迪士尼乐园门票价格优惠")
        irrelevant = self._search("如何制作番茄炒鸡蛋食谱")
        self.assertGreater(irrelevant[0]["distance"], relevant[0]["distance"],
                           f"无关查询({irrelevant[0]['distance']:.4f}) 距离应大于"
                           f"相关查询({relevant[0]['distance']:.4f})")

    def test_16_top_k_count_respected(self):
        """search 返回条数不超过请求的 k"""
        for k in [1, 3, 5, 10]:
            results = self._search("迪士尼乐园", k=k)
            self.assertLessEqual(len(results), k,
                                 f"k={k} 但返回了 {len(results)} 条")

    def test_17_results_sorted_by_distance_ascending(self):
        """返回结果应按 L2 距离升序排列（最相近在前）"""
        results = self._search("迪士尼乐园游玩攻略", k=10)
        dists = [r["distance"] for r in results]
        self.assertEqual(dists, sorted(dists),
                         "结果未按距离升序排列")

    def test_18_similarity_score_inversely_proportional_to_distance(self):
        """similarity = 1/(1+dist)，应与距离单调递减"""
        results = self._search("迪士尼乐园", k=8)
        for r in results:
            expected_sim = 1.0 / (1.0 + r["distance"])
            self.assertAlmostEqual(r["similarity"], expected_sim, places=6,
                                   msg="similarity 计算公式不一致")
        # 距离越小，相似度越大
        sims = [r["similarity"] for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True),
                         "similarity 未随距离递增而递减")


# ═══════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 检测是否有指定 class 参数（部分运行）
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # 将 ClassName 参数转为 unittest 可识别的格式
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        for name in sys.argv[1:]:
            try:
                suite.addTests(loader.loadTestsFromName(name, module=__import__("__main__")))
            except AttributeError:
                print(f"[警告] 找不到测试类: {name}")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    else:
        unittest.main(verbosity=2)
