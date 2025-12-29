"""Gemini API client for AI-powered features."""

import os
from typing import Optional
import google.generativeai as genai


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.enabled = bool(self.api_key)

        if self.enabled:
            genai.configure(api_key=self.api_key)
            # Use lightweight flash model for cost efficiency
            self.model = genai.GenerativeModel('gemini-3-flash-preview')



    def generate_hint(
        self,
        problem_title: str,
        problem_description: str,
        setup_code: str,
        user_code: Optional[str] = None,
        language: str = "ja"
    ) -> Optional[str]:
        """Generate a helpful hint for solving the problem.

        Args:
            problem_title: Problem title
            problem_description: Problem description
            setup_code: Setup code
            user_code: User's attempted code (if any)
            language: Language for hint ("ja" or "en")

        Returns:
            Hint string, or None if disabled
        """
        if not self.enabled:
            return None

        user_attempt = f"\n\nUser's current attempt:\n```python\n{user_code}\n```" if user_code else ""

        prompt = f"""Provide a helpful hint for this PyTorch tensor problem. Don't give away the full solution code immediately, but guide the learner.

Problem: {problem_title}
Description: {problem_description}

Setup Code:
```python
{setup_code}
```
{user_attempt}

Provide 2-3 hints that:
1. Guide toward the right PyTorch functions/operations
2. Explain relevant tensor concepts
3. **Important**: Briefly suggest an alternative approach or solution if one exists ("別解" / "Alternative").

Format your response using Markdown (use lists, bold text, code blocks).
Respond in {"Japanese" if language == "ja" else "English"}.
Be encouraging and educational.
"""

        response = self.model.generate_content(prompt)
        return response.text

    def suggest_improvements(
        self,
        user_code: str,
        expected_code: str,
        language: str = "ja"
    ) -> Optional[str]:
        """Suggest improvements to user's correct solution.

        Args:
            user_code: User's working solution
            expected_code: Expected/optimal solution
            language: Language for suggestions ("ja" or "en")

        Returns:
            Improvement suggestions string, or None if disabled
        """
        if not self.enabled:
            return None

        prompt = f"""Compare these two PyTorch solutions and suggest improvements to the user's code.

User's Solution:
```python
{user_code}
```

Expected Solution:
```python
{expected_code}
```

Provide constructive feedback on:
1. Code efficiency (is there a faster/more efficient approach?)
2. Readability and PyTorch best practices
3. Alternative methods they might consider
4. What they did well

Respond in {"Japanese" if language == "ja" else "English"}.
Be positive and educational - the user's code works, so focus on learning opportunities.
"""

        response = self.model.generate_content(prompt)
        return response.text
