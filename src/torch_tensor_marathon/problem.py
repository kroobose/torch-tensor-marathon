"""Problem data structure and problem bank management."""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
import torch


@dataclass
class ProblemCase:
    """Represents a specific variation/test case of a problem."""
    name: str  # e.g. "Case 1", "v2", or specific name
    description_ja: str
    description_en: str
    setup_code: str
    solution_code: str
    hint_ja: str = ""
    hint_en: str = ""

    # Optional per-case override
    test_function: Callable[[torch.Tensor], torch.Tensor] | None = None


@dataclass
class Problem:
    """Represents a single tensor manipulation problem containing multiple variations."""

    id: str  # Unique identifier (e.g., "reshape_001")
    category: str  # Category ID (e.g., "reshape_permute")
    difficulty: str  # "beginner", "intermediate", "advanced", "expert"

    # Multilingual content (Main generic title)
    title_ja: str
    title_en: str

    # Primary Case (Case 1) / Defaults
    # kept for backward compatibility with existing problem definitions
    description_ja: str = ""
    description_en: str = ""
    hint_ja: str = ""
    hint_en: str = ""
    setup_code: str = ""
    solution_code: str = ""

    # Test function
    test_function: Callable[[torch.Tensor], torch.Tensor] | None = None
    expected_shape: tuple | None = None

    # Tags for additional filtering
    tags: List[str] = field(default_factory=list)

    # The list of variations
    cases: List[ProblemCase] = field(default_factory=list)

    def __post_init__(self):
        """Ensure consistency between flat fields and cases."""
        # If no cases defined, create the first case from flat fields
        if not self.cases and self.setup_code:
            self.cases.append(ProblemCase(
                name="Main",
                description_ja=self.description_ja,
                description_en=self.description_en,
                setup_code=self.setup_code,
                solution_code=self.solution_code,
                hint_ja=self.hint_ja,
                hint_en=self.hint_en,
                test_function=self.test_function
            ))
        # If cases ARE defined but flat fields are empty (future usage), populate flat fields from case 0
        elif self.cases and not self.setup_code:
            self.description_ja = self.cases[0].description_ja
            self.description_en = self.cases[0].description_en
            self.hint_ja = self.cases[0].hint_ja
            self.hint_en = self.cases[0].hint_en
            self.setup_code = self.cases[0].setup_code
            self.solution_code = self.cases[0].solution_code

    def get_title(self, lang: str) -> str:
        """Get title in specified language."""
        return self.title_ja if lang == "ja" else self.title_en

    def get_description(self, lang: str, case_idx: int = 0) -> str:
        """Get description in specified language for a specific case."""
        if 0 <= case_idx < len(self.cases):
            return self.cases[case_idx].description_ja if lang == "ja" else self.cases[case_idx].description_en
        return self.description_ja if lang == "ja" else self.description_en

    def get_hint(self, lang: str, case_idx: int = 0) -> str:
        """Get hint in specified language for a specific case."""
        if 0 <= case_idx < len(self.cases):
            return self.cases[case_idx].hint_ja if lang == "ja" else self.cases[case_idx].hint_en
        return self.hint_ja if lang == "ja" else self.hint_en

    def get_full_setup_code(self, case_idx: int = 0) -> str:
        """Get the complete setup code for a specific case."""
        if 0 <= case_idx < len(self.cases):
            return self.cases[case_idx].setup_code
        return self.setup_code

    def get_complete_solution(self, case_idx: int = 0) -> str:
        """Get the complete solution code (setup + solution) for a specific case."""
        if 0 <= case_idx < len(self.cases):
            return self.cases[case_idx].setup_code + "\n" + self.cases[case_idx].solution_code
        return self.setup_code + "\n" + self.solution_code


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
