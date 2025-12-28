"""Example usage script demonstrating programmatic access to problems."""

from torch_tensor_marathon.problems import initialize_problems
from torch_tensor_marathon.problem import problem_bank
from torch_tensor_marathon.checker import CorrectnessChecker
import torch

def main():
    # Initialize the problem bank
    initialize_problems()

    print(f"📚 Total problems loaded: {len(problem_bank)}")
    print(f"📁 Categories: {len(problem_bank.get_categories())}\n")

    # Show category statistics
    for cat in problem_bank.get_categories():
        stats = problem_bank.get_category_stats(cat)
        print(f"{cat}: {stats['total']} problems")
        print(f"  Beginner: {stats['beginner']}, Intermediate: {stats['intermediate']}, "
              f"Advanced: {stats['advanced']}, Expert: {stats['expert']}")

    print("\n" + "="*60)
    print("Example: Testing a problem programmatically")
    print("="*60 + "\n")

    # Get a specific problem
    problem = problem_bank.get_problem("reshape_001")

    if problem:
        print(f"Problem: {problem.get_title('en')}")
        print(f"Description: {problem.get_description('en')}")
        print(f"Difficulty: {problem.difficulty}\n")

        print("Setup Code:")
        print(problem.setup_code)
        print()

        # Test the solution
        checker = CorrectnessChecker()

        # Correct solution
        print("Testing correct solution:", problem.solution_code)
        result = checker.check_solution(
            setup_code=problem.setup_code,
            user_code=problem.solution_code,
            solution_code=problem.solution_code,
        )

        if result.is_correct:
            print(f"✅ {result.message}")
        else:
            print(f"❌ {result.message}")
    else:
        print("Problem not found!")


if __name__ == "__main__":
    main()
