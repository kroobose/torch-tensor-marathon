
import torch
from torch_tensor_marathon.problem import Problem, ProblemCase
from torch_tensor_marathon.checker import CorrectnessChecker

def test_multi_case_checker():
    print("Testing CorrectnessChecker with multi-case problem...")

    # Define a dummy problem with 2 cases
    case1 = ProblemCase(
        name="Case 1 (N=1)",
        description_ja="", description_en="",
        setup_code="x = torch.tensor([1.0])\nN = 1.0",
        solution_code="result = x + N"
    )

    case2 = ProblemCase(
        name="Case 2 (N=2)",
        description_ja="", description_en="",
        setup_code="x = torch.tensor([1.0])\nN = 2.0",
        solution_code="result = x + N"
    )

    problem = Problem(
        id="test_multi",
        category="test",
        difficulty="beginner",
        title_ja="Test",
        title_en="Test",
        cases=[case1, case2],
        tags=[]
    )

    checker = CorrectnessChecker()

    # 1. Test Correct Generic Solution
    print("\n--- Test 1: Correct Generic Solution ---")
    user_code_correct = "result = x + N"
    result = checker.check_problem(problem, user_code_correct)

    print(f"Overall Correct: {result.is_correct}")
    if result.case_results:
        for cr in result.case_results:
            print(f"  {cr['name']}: {cr['is_correct']} ({cr['message']})")

    assert result.is_correct
    assert len(result.case_results) == 2
    assert result.case_results[0]['is_correct']
    assert result.case_results[1]['is_correct']

    # 2. Test Partial Failure (Hardcoded for Case 1)
    print("\n--- Test 2: Partial Failure (Hardcoded) ---")
    # This code works for Case 1 (1+1=2) but fails Case 2 (1+2!=2)
    user_code_partial = "result = torch.tensor([2.0])"
    result = checker.check_problem(problem, user_code_partial)

    print(f"Overall Correct: {result.is_correct}")
    if result.case_results:
        for cr in result.case_results:
            print(f"  {cr['name']}: {cr.get('is_correct')} ({cr.get('message')})")

    assert not result.is_correct
    assert result.case_results[0]['is_correct'] # 2.0 matches Case 1
    assert not result.case_results[1]['is_correct'] # 2.0 != 3.0 (Case 2)

    # 3. Test Syntax Error
    print("\n--- Test 3: Syntax Error ---")
    user_code_syntax = "result = "
    result = checker.check_problem(problem, user_code_syntax)
    print(f"Overall Correct: {result.is_correct}")
    print(f"Error Type: {result.error_type}")
    if result.case_results:
         print(f"Case 0 Error: {result.case_results[0].get('error_type')}")

    assert not result.is_correct
    # If error_type is None, that means it wasn't propagated.
    # We accept "execution" or "syntax" (if parsed).
    assert result.error_type in ["syntax", "execution"]

    print("\nPASSED: All multi-case checker tests passed.")

if __name__ == "__main__":
    test_multi_case_checker()
