"""Problem data structure and problem bank management."""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
import torch


@dataclass
class Problem:
    """Represents a single tensor manipulation problem."""

    id: str  # Unique identifier (e.g., "reshape_001")
    category: str  # Category ID (e.g., "reshape_permute")
    difficulty: str  # "beginner", "intermediate", "advanced", "expert"

    # Multilingual content
    title_ja: str
    title_en: str
    description_ja: str
    description_en: str
    hint_ja: str = ""
    hint_en: str = ""

    # Code template
    setup_code: str = ""  # Code to run before user's solution
    template_code: str = ""  # Template for user to fill in
    solution_code: str = ""  # Expected solution

    # Test function
    test_function: Callable[[torch.Tensor], torch.Tensor] | None = None
    expected_shape: tuple | None = None

    # Tags for additional filtering
    tags: List[str] = field(default_factory=list)

    def get_title(self, lang: str) -> str:
        """Get title in specified language."""
        return self.title_ja if lang == "ja" else self.title_en

    def get_description(self, lang: str) -> str:
        """Get description in specified language."""
        return self.description_ja if lang == "ja" else self.description_en

    def get_hint(self, lang: str) -> str:
        """Get hint in specified language."""
        return self.hint_ja if lang == "ja" else self.hint_en

    def get_full_setup_code(self) -> str:
        """Get the complete setup code."""
        # No need to add imports since checker provides torch in namespace
        return self.setup_code

    def get_complete_solution(self) -> str:
        """Get the complete solution code (setup + solution)."""
        return self.get_full_setup_code() + self.solution_code


class ProblemBank:
    """Manages the collection of all problems."""

    def __init__(self):
        self.problems: List[Problem] = []
        self._problems_by_id: Dict[str, Problem] = {}
        self._problems_by_category: Dict[str, List[Problem]] = {}

    def add_problem(self, problem: Problem):
        """Add a problem to the bank."""
        self.problems.append(problem)
        self._problems_by_id[problem.id] = problem

        if problem.category not in self._problems_by_category:
            self._problems_by_category[problem.category] = []
        self._problems_by_category[problem.category].append(problem)

    def add_problems(self, problems: List[Problem]):
        """Add multiple problems to the bank."""
        for problem in problems:
            self.add_problem(problem)

    def get_problem(self, problem_id: str) -> Problem | None:
        """Get a problem by ID."""
        return self._problems_by_id.get(problem_id)

    def get_problems_by_category(self, category: str) -> List[Problem]:
        """Get all problems in a category."""
        return self._problems_by_category.get(category, [])

    def get_problems_by_difficulty(self, difficulty: str) -> List[Problem]:
        """Get all problems of a certain difficulty."""
        return [p for p in self.problems if p.difficulty == difficulty]

    def get_problems_by_tag(self, tag: str) -> List[Problem]:
        """Get all problems with a specific tag."""
        return [p for p in self.problems if tag in p.tags]

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self._problems_by_category.keys())

    def get_category_stats(self, category: str) -> Dict[str, int]:
        """Get statistics for a category."""
        problems = self.get_problems_by_category(category)
        return {
            "total": len(problems),
            "beginner": len([p for p in problems if p.difficulty == "beginner"]),
            "intermediate": len([p for p in problems if p.difficulty == "intermediate"]),
            "advanced": len([p for p in problems if p.difficulty == "advanced"]),
            "expert": len([p for p in problems if p.difficulty == "expert"]),
        }

    def __len__(self) -> int:
        """Return total number of problems."""
        return len(self.problems)


# Global problem bank instance
problem_bank = ProblemBank()
