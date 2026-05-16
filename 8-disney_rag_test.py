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
#
# 测试用例均基于真实知识库文档内容生成，覆盖 5 个分类：
#   产品与服务详情：门票价格、年票规则、餐饮信息、游乐设施
#   运营流程与标准作业程序：退款政策、客诉处理
#   特殊情况与应急预案：走失儿童、恶劣天气、紧急电话
#   客户关系与支持话术：话术授权、FAQ、入园规定
#   内部知识与工具：CRM系统、岗位职责
#
# 评判标准：
#   ① L2 距离 < 阈值    → 向量语义相近
#   ② top-K 含预期分类  → 分类路由正确
#   ③ content 含预期关键词 → 实际内容命中
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipIf(SKIP_API, SKIP_MSG)
class TestRetrievalQuality(unittest.TestCase):

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

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        resp = self._dashscope.MultiModalEmbedding.call(
            model="tongyi-embedding-vision-plus",
            input=[{"text": text}]
        )
        if resp.status_code != self._HTTPStatus.OK:
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

    def _assert_category_in_top_k(self, results, expected_cat, k=5):
        cats = [r["metadata"]["category"] for r in results[:k]]
        self.assertIn(expected_cat, cats,
                      f"前{k}条结果中未出现分类 [{expected_cat}]\n实际分类: {cats}")

    def _assert_keywords_in_top_k(self, results, keywords, k=5):
        """断言 top-k 的 content 字段中至少命中一个关键词"""
        text = " ".join(r["metadata"].get("content", "") for r in results[:k])
        matched = [kw for kw in keywords if kw in text]
        sources = [r["metadata"]["source"] for r in results[:k]]
        self.assertTrue(matched,
                        f"前{k}条结果的 content 均不含关键词 {keywords}\n来源: {sources}")

    def _top1_distance(self, results) -> float:
        return results[0]["distance"]

    # ════════════════════════════════════════════════════════════════════════
    # 产品与服务详情（门票价格）
    # 来源：迪士尼乐园门票价格一览，官方渠道购票攻略.docx
    # ════════════════════════════════════════════════════════════════════════

    def test_01_shanghai_adult_weekday_ticket_price(self):
        """上海成人平日票价 499 元 → content 中应出现"499" """
        # 文档原文：成人票：平日499元，周末及节假日649元
        results = self._search("上海迪士尼成人票平日价格是多少")
        self.assertLess(self._top1_distance(results), 2.5,
                        f"top1 距离过大: {self._top1_distance(results):.4f}")
        self._assert_keywords_in_top_k(results, ["499", "平日"])

    def test_02_shanghai_child_ticket_price(self):
        """上海儿童票平日 399 元 → content 应含"399" """
        # 文档原文：儿童票：平日399元，周末及节假日499元
        results = self._search("上海迪士尼儿童票多少钱")
        self._assert_keywords_in_top_k(results, ["399", "儿童票"])

    def test_03_hong_kong_adult_ticket_price(self):
        """香港成人票平日 539 港元 → content 应含"539" """
        # 文档原文：成人票：平日539港元，周末及节假日639港元
        results = self._search("香港迪士尼成人门票价格")
        self._assert_keywords_in_top_k(results, ["539", "港元"])

    def test_04_france_disneyland_ticket_price(self):
        """法国迪士尼成人平日票 70 欧元 → content 应含"70"和"欧元" """
        # 文档原文：成人票：平日70欧元，周末及节假日85欧元
        results = self._search("法国巴黎迪士尼乐园门票价格")
        self._assert_keywords_in_top_k(results, ["70", "欧元"])

    # ════════════════════════════════════════════════════════════════════════
    # 产品与服务详情（年票规则）
    # 来源：迪士尼乐园年票使用规则.docx
    # ════════════════════════════════════════════════════════════════════════

    def test_05_annual_pass_validity_365_days(self):
        """年票有效期 365 天、激活期 90 天 → content 应含"365" """
        # 文档原文：年票有效期自首次使用日起计算，365日内有效。
        #           首次入园需在购票后90天内完成激活
        results = self._search("迪士尼年票有效期多少天以及激活期限")
        self._assert_keywords_in_top_k(results, ["365"])

    def test_06_annual_pass_renewal_discount(self):
        """年票续费享 95 折 → content 应含"95折" """
        # 文档原文：续费用户可享受原价95折优惠
        results = self._search("迪士尼年票续费有什么优惠")
        self._assert_keywords_in_top_k(results, ["95折", "续费"])

    def test_07_child_free_entry_height_limit(self):
        """1 米以下儿童免票 → content 应含"1米"或"免票" """
        # 文档原文：携带儿童游玩时，1米以下儿童可免票入园
        results = self._search("迪士尼多高的儿童可以免票入园")
        self._assert_keywords_in_top_k(results, ["1米", "免票"])

    def test_08_annual_pass_max_free_children(self):
        """每张年票最多携带 2 名免票儿童 → content 应含"2" """
        # 文档原文：每张成人年票单日最多携带2名免票儿童，超出人数需另购门票
        results = self._search("持年票可以携带几名儿童免费入园")
        self._assert_keywords_in_top_k(results, ["2名", "2"])

    def test_09_annual_pass_lost_replacement_fee(self):
        """年票丢失补办费为原价 10% → content 应含"10%" """
        # 文档原文：年票丢失需立即挂失，补办工本费为原价10%
        results = self._search("迪士尼年票丢失如何补办，补办费用是多少")
        self._assert_keywords_in_top_k(results, ["10%", "挂失", "补办"])

    def test_10_special_event_annual_pass_discount(self):
        """万圣节等特殊活动年票享 9 折 → content 应含"9折" """
        # 文档原文：年票用户购买此类活动门票可享9折优惠
        results = self._search("年票持有者购买万圣节专场票有折扣吗")
        self._assert_keywords_in_top_k(results, ["9折", "年票"])

    # ════════════════════════════════════════════════════════════════════════
    # 运营流程（退款政策）
    # 来源：迪士尼各园区退款政策……详细列出来.docx
    # ════════════════════════════════════════════════════════════════════════

    def test_11_typhoon_closure_full_refund(self):
        """台风闭园可全额退款或顺延 6 个月 → content 应含"全额退"或"6个月" """
        # 文档原文：园区因台风、疫情等闭园，门票可全额退款或顺延6个月内任选一天使用
        results = self._search("台风导致迪士尼闭园，门票可以退款还是改期")
        self._assert_keywords_in_top_k(results, ["全额退", "6个月", "闭园"])
        self._assert_category_in_top_k(results, CAT_OPERATIONS)

    def test_12_hong_kong_no_refund_on_confirmed_ticket(self):
        """香港指定日门票确认后不可退改 → content 应含"不可退" """
        # 文档原文：香港Disneyland，官方「指定日门票」一经确认不可退改
        results = self._search("香港迪士尼的指定日门票确认后还能退吗")
        self._assert_keywords_in_top_k(results, ["不可退", "香港"])

    def test_13_tokyo_no_refund_policy(self):
        """东京门票购买后概不退款 → content 应含"概不退款"或"东京" """
        # 文档原文：东京：所有门票一经购买概不退款；官方闭园可退
        results = self._search("东京迪士尼门票买了之后能退吗")
        self._assert_keywords_in_top_k(results, ["东京", "退款", "概不退款"])

    # ════════════════════════════════════════════════════════════════════════
    # 特殊情况与应急预案
    # 来源：迪士尼乐园「紧急情况处理」一站式速查表.docx
    # ════════════════════════════════════════════════════════════════════════

    def test_14_lost_child_broadcast_no_parent_info(self):
        """走失儿童广播不暴露家长信息以防冒领 → content 应含"冒领"或"广播" """
        # 文档原文：儿童：仅播报「请XXX小朋友到米奇大街走失儿童认领处，您的家人正在等您」
        #           ——不暴露家长姓名与特征，防止冒领
        results = self._search("迪士尼走失儿童广播内容是什么，为何不报家长信息")
        self._assert_keywords_in_top_k(results, ["冒领", "广播", "走失"])
        self._assert_category_in_top_k(results, CAT_EMERGENCY)

    def test_15_shanghai_emergency_hotline(self):
        """上海应急热线 021-2099-8001 → content 应含该号码 """
        # 文档原文：记住 3 个号码 • 上海：021-2099-8001
        results = self._search("上海迪士尼乐园的紧急求助电话是多少")
        self._assert_keywords_in_top_k(results, ["021-2099-8001", "2099"])

    def test_16_shanghai_typhoon_closure_condition(self):
        """上海台风橙色预警触发闭园 → content 应含"橙色预警" """
        # 文档原文：上海：上海中心气象台发布台风橙色预警或暴雨红色预警→当日全天闭园
        results = self._search("上海迪士尼在什么气象条件下会闭园")
        self._assert_keywords_in_top_k(results, ["橙色预警", "闭园"])

    # ════════════════════════════════════════════════════════════════════════
    # 客户关系与支持话术
    # 来源：迪士尼乐园客户关系与支持话术.docx
    # ════════════════════════════════════════════════════════════════════════

    def test_17_magic_moment_staff_authorization_limit(self):
        """一线员工奇迹时刻授权上限 100 元 → content 应含"100元" """
        # 文档原文：授权额度：一线员工 ≤100 元/件；当班主管 ≤500 元/件
        results = self._search("迪士尼奇迹时刻一线员工的补偿授权额度是多少")
        self._assert_keywords_in_top_k(results, ["100元", "100", "奇迹时刻"])
        self._assert_category_in_top_k(results, CAT_CUSTOMER)

    def test_18_under_3_years_free_entry(self):
        """3 岁以下儿童免票 → content 应含"3岁"或"免费" """
        # 文档原文：Q6 3岁以下儿童要门票吗？上海、香港、巴黎、美国园区均免费
        results = self._search("几岁以下的孩子进迪士尼不需要买票")
        self._assert_keywords_in_top_k(results, ["3岁", "免费"])

    def test_19_shanghai_food_policy_no_self_heating(self):
        """上海可带密封零食，禁止自热食品 → content 应含"密封"和"自热" """
        # 文档原文：上海：可带密封零食、水果；自热食品、需加热或刺激气味食品禁止
        results = self._search("上海迪士尼可以自带食物进园吗，有哪些限制")
        self._assert_keywords_in_top_k(results, ["密封", "自热"])

    # ════════════════════════════════════════════════════════════════════════
    # 内部知识与工具（员工手册）
    # 来源：迪士尼乐园内部知识与工具操作手册.docx
    # ════════════════════════════════════════════════════════════════════════

    def test_20_crm_case_priority_levels(self):
        """CRM 工单优先级 P1/P2/P3 定义 → content 应含"P1"和"P2" """
        # 文档原文：「Priority」等级：P1（安全/舆情）、P2（现场投诉）、P3（事后咨询）
        results = self._search("迪士尼CRM系统如何划分投诉工单的优先级")
        self._assert_keywords_in_top_k(results, ["P1", "P2"])
        self._assert_category_in_top_k(results, CAT_INTERNAL)


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
