
import sys
import os
import torch
import torch.nn.functional as F

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from torch_tensor_marathon.problems import initialize_problems

def verify_problems():
    bank = initialize_problems()
    problems = bank.problems

    print(f"Total problems found: {len(problems)}")

    total_setup_errors = 0
    total_solution_errors = 0

    for problem in problems:
        # print(f"Verifying {problem.id}...", end="")

        # Determine cases to check
        cases_to_check = problem.cases if problem.cases else []
        if not cases_to_check and problem.setup_code:
             # Should have been handled by post_init, but just in case
             pass

        if not cases_to_check:
             print(f"WARNING: No cases found for {problem.id}")
             continue

        for i, case in enumerate(cases_to_check):
            case_name = case.name if hasattr(case, "name") else f"Case {i+1}"

            # Check setup execution
            try:
                setup_globals = {"torch": torch, "F": F}
                exec(case.setup_code, setup_globals)
            except Exception as e:
                print(f"ERROR in {problem.id} [{case_name}] SETUP: {e}")
                total_setup_errors += 1
                continue

            # Check solution execution
            try:
                solution_globals = setup_globals.copy()
                exec(case.solution_code, solution_globals)
                if "result" not in solution_globals and "output" not in solution_globals:
                    print(f"ERROR in {problem.id} [{case_name}] SOLUTION: result/output variable not found")
                    total_solution_errors += 1
                else:
                    pass
            except Exception as e:
                print(f"ERROR in {problem.id} [{case_name}] SOLUTION: {e}")
                total_solution_errors += 1

    print("-" * 20)
    print(f"Verification complete.")
    print(f"Setup errors: {total_setup_errors}")
    print(f"Solution errors: {total_solution_errors}")

if __name__ == "__main__":
    verify_problems()
