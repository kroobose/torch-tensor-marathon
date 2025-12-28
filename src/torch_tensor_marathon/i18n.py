"""Internationalization support for Japanese and English."""

from typing import Dict
from torch_tensor_marathon.config import Language

# Translation dictionaries
TRANSLATIONS: Dict[str, Dict[Language, str]] = {
    # UI Elements
    "app_title": {
        "ja": "🏃 PyTorch テンソルマラソン",
        "en": "🏃 PyTorch Tensor Marathon",
    },
    "welcome": {
        "ja": "テンソル操作の練習へようこそ！",
        "en": "Welcome to Tensor Operation Practice!",
    },
    "select_category": {
        "ja": "カテゴリを選択してください:",
        "en": "Select a category:",
    },
    "select_problem": {
        "ja": "問題を選択してください:",
        "en": "Select a problem:",
    },
    "difficulty": {
        "ja": "難易度",
        "en": "Difficulty",
    },
    "category": {
        "ja": "カテゴリ",
        "en": "Category",
    },
    "problem": {
        "ja": "問題",
        "en": "Problem",
    },
    "your_solution": {
        "ja": "あなたの解答",
        "en": "Your Solution",
    },
    "result": {
        "ja": "結果",
        "en": "Result",
    },
    "correct": {
        "ja": "✅ 正解！",
        "en": "✅ Correct!",
    },
    "incorrect": {
        "ja": "❌ 不正解",
        "en": "❌ Incorrect",
    },
    "shape_error": {
        "ja": "形状エラー: 期待値 {expected}、実際の値 {actual}",
        "en": "Shape Error: Expected {expected}, got {actual}",
    },
    "value_error": {
        "ja": "値エラー: 出力テンソルの値が一致しません",
        "en": "Value Error: Output tensor values do not match",
    },
    "execution_error": {
        "ja": "実行エラー: {error}",
        "en": "Execution Error: {error}",
    },

    # Categories
    "cat_reshape_permute": {
        "ja": "🔄 Reshape & Permute",
        "en": "🔄 Reshape & Permute",
    },
    "cat_indexing_slicing": {
        "ja": "🎯 Indexing & Slicing",
        "en": "🎯 Indexing & Slicing",
    },
    "cat_broadcasting": {
        "ja": "📡 Broadcasting & Arithmetic",
        "en": "📡 Broadcasting & Arithmetic",
    },
    "cat_gather_scatter": {
        "ja": "🎲 Gather & Scatter",
        "en": "🎲 Gather & Scatter",
    },
    "cat_einsum": {
        "ja": "∑ Einstein Summation",
        "en": "∑ Einstein Summation",
    },
    "cat_stacking_splitting": {
        "ja": "📚 Stacking & Splitting",
        "en": "📚 Stacking & Splitting",
    },
    "cat_advanced_ops": {
        "ja": "⚡ Advanced Operations",
        "en": "⚡ Advanced Operations",
    },
    "cat_dl_applications": {
        "ja": "🧠 Deep Learning Applications",
        "en": "🧠 Deep Learning Applications",
    },

    # Difficulty levels
    "beginner": {
        "ja": "初級",
        "en": "Beginner",
    },
    "intermediate": {
        "ja": "中級",
        "en": "Intermediate",
    },
    "advanced": {
        "ja": "上級",
        "en": "Advanced",
    },
    "expert": {
        "ja": "エキスパート",
        "en": "Expert",
    },

    # Instructions
    "input_tensor": {
        "ja": "入力テンソル",
        "en": "Input Tensor",
    },
    "goal": {
        "ja": "目標",
        "en": "Goal",
    },
    "hint": {
        "ja": "ヒント",
        "en": "Hint",
    },
    "expected_shape": {
        "ja": "期待される形状",
        "en": "Expected Shape",
    },
}


def t(key: str, lang: Language, **kwargs) -> str:
    """
    Translate a key to the specified language.

    Args:
        key: Translation key
        lang: Target language ('ja' or 'en')
        **kwargs: Format arguments for string interpolation

    Returns:
        Translated string
    """
    translation = TRANSLATIONS.get(key, {}).get(lang, key)
    if kwargs:
        return translation.format(**kwargs)
    return translation


def get_category_name(category_id: str, lang: Language) -> str:
    """Get the display name for a category."""
    return t(f"cat_{category_id}", lang)
