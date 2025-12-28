"""Configuration management for the Tensor Marathon."""

import os
from pathlib import Path
from typing import Literal

Language = Literal["ja", "en"]


class Config:
    """Global configuration for the application."""

    def __init__(self):
        self.language: Language = "en"  # Default to English
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
        self.use_gemini: bool = self.gemini_api_key is not None

        # Difficulty levels
        self.difficulty_levels = ["beginner", "intermediate", "advanced", "expert"]

        # Data directory for caching
        self.cache_dir = Path.home() / ".cache" / "torch_tensor_marathon"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # User progress file
        self.progress_file = Path.cwd() / "user_progress.json"

    def set_language(self, lang: Language):
        """Set the UI language."""
        if lang not in ["ja", "en"]:
            raise ValueError(f"Unsupported language: {lang}. Use 'ja' or 'en'.")
        self.language = lang

    def set_gemini_api_key(self, api_key: str):
        """Set Gemini API key."""
        self.gemini_api_key = api_key
        self.use_gemini = True


# Global config instance
config = Config()
