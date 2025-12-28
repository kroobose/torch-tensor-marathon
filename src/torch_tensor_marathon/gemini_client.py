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

    def explain_solution(
        self,
        problem_title: str,
        problem_description: str,
        setup_code: str,
        solution_code: str,
        language: str = "ja"
    ) -> Optional[str]:
        """Generate an AI-powered explanation of the solution.

        Args:
            problem_title: Problem title
            problem_description: Problem description
            setup_code: Setup code
            solution_code: Solution code to explain
            language: Language for explanation ("ja" or "en")

        Returns:
            Detailed explanation string, or None if disabled
        """
        if not self.enabled:
            return None

        prompt = f"""Explain this PyTorch tensor solution in detail for a learner.

Problem: {problem_title}
Description: {problem_description}

Setup Code:
```python
{setup_code}
```

Solution:
```python
{solution_code}
```

Provide a clear, educational explanation that covers:
1. What the solution does step-by-step
2. Why this approach works
3. Key PyTorch concepts used
4. Any potential pitfalls or alternatives

Respond in {"Japanese" if language == "ja" else "English"}.
Keep it concise but thorough (3-5 paragraphs).
"""

        response = self.model.generate_content(prompt)
        return response.text

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

        prompt = f"""Provide a helpful hint for this PyTorch tensor problem. Don't give away the full solution, but guide the learner in the right direction.

Problem: {problem_title}
Description: {problem_description}

Setup Code:
```python
{setup_code}
```{user_attempt}

Provide 2-3 hints that:
1. Guide toward the right PyTorch functions/operations
2. Explain relevant tensor concepts
3. Suggest the general approach without revealing the exact code

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
