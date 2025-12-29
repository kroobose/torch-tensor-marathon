"""Correctness checker for validating user solutions."""

from dataclasses import dataclass
from typing import Dict, Any, List
import torch
import traceback
import sys
from io import StringIO


@dataclass
class CheckResult:
    """Result of checking a user's solution."""

    is_correct: bool
    message: str
    error_type: str | None = None  # "shape", "value", "execution", None
    expected_shape: tuple | None = None
    actual_shape: tuple | None = None
    execution_output: str = ""  # Captured stdout
    case_results: List[dict] | None = None  # Detailed per-case results
    actual_values: str | None = None  # String representation of user's result tensor
    expected_values: str | None = None  # String representation of expected tensor

    def __bool__(self) -> bool:
        return self.is_correct


class CorrectnessChecker:
    """Validates user solutions against expected outputs."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        """
        Initialize the checker.

        Args:
            rtol: Relative tolerance for torch.allclose
            atol: Absolute tolerance for torch.allclose
        """
        self.rtol = rtol
        self.atol = atol

    def check_problem(self, problem: Any, user_code: str, case_name: str | None = None) -> CheckResult:
        """
        Check solution against all cases in a problem, or a specific case.

        Args:
            problem: Problem object containing cases
            user_code: User's solution code
            case_name: Optional specific case name to check

        Returns:
            Aggregate CheckResult or specific CheckResult
        """
        results = []
        all_correct = True
        aggregate_messages = []
        first_error_result = None

        # Determine cases to check
        cases = problem.cases
        if case_name:
            # Filter for the specific case
            cases = [c for c in cases if c.name == case_name]
            if not cases:
                return CheckResult(False, f"Case '{case_name}' not found in problem.")

        for i, case in enumerate(cases):
            current_case_name = case.name if hasattr(case, "name") else f"Case {i+1}"

            # Check this specific case
            res = self.check_solution(
                setup_code=case.setup_code,
                user_code=user_code,
                solution_code=case.solution_code
            )

            # Store result
            case_result = {
                "name": current_case_name,
                "is_correct": res.is_correct,
                "message": res.message,
                "error_type": res.error_type,
                "execution_output": res.execution_output,
                "expected_shape": res.expected_shape,
                "actual_shape": res.actual_shape
            }
            results.append(case_result)

            # Update aggregate status
            if res.is_correct:
                aggregate_messages.append(f"✅ {current_case_name}: Passed")
            else:
                all_correct = False
                aggregate_messages.append(f"❌ {current_case_name}: Failed - {res.message}")
                if first_error_result is None:
                    first_error_result = res

        # Construct final result
        if all_correct:
            # If we were checking a specific case, return its details directly
            if case_name and len(results) == 1:
                base = res # From the loop
                return CheckResult(
                    is_correct=True,
                    message=f"✅ {case_name}: Passed",
                    expected_shape=base.expected_shape,
                    actual_shape=base.actual_shape,
                    execution_output=base.execution_output,
                    case_results=results
                )

            return CheckResult(
                is_correct=True,
                message="✅ All cases passed!\n" + "\n".join(aggregate_messages),
                case_results=results,
                execution_output=results[0]["execution_output"] if results else ""
            )
        else:
            # Use details from the first error for top-level metadata
            base = first_error_result if first_error_result is not None else CheckResult(False, "Unknown error")

            error_message = f"❌ Some cases failed.\n" + "\n".join(aggregate_messages)
            if case_name:
                error_message = f"❌ {case_name}: Failed\n{base.message}"

            return CheckResult(
                is_correct=False,
                message=error_message,
                error_type=base.error_type,
                expected_shape=base.expected_shape,
                actual_shape=base.actual_shape,
                execution_output=base.execution_output,
                case_results=results
            )

    def check_solution(
        self,
        setup_code: str,
        user_code: str,
        solution_code: str,
        expected_shape: tuple | None = None,
    ) -> CheckResult:
        """
        Check if user's code produces the correct output.

        Args:
            setup_code: Code to set up the problem (creates input tensors, etc.)
            user_code: User's solution code
            solution_code: Expected solution code
            expected_shape: Expected output shape (optional)

        Returns:
            CheckResult with validation status and details
        """
        # Execute expected solution
        try:
            torch.manual_seed(42)  # Ensure deterministic setup
            expected_namespace = self._execute_code(
                setup_code + "\n" + solution_code
            )
        except Exception as e:
            return CheckResult(
                is_correct=False,
                message=f"Error in expected solution: {str(e)}",
                error_type="execution",
            )

        # Execute user solution
        try:
            torch.manual_seed(42)  # Ensure deterministic setup
            user_namespace, stdout = self._execute_code_with_output(
                setup_code + "\n" + user_code
            )
        except Exception as e:
            return CheckResult(
                is_correct=False,
                message=f"Execution Error: {str(e)}\n\n{traceback.format_exc()}",
                error_type="execution",
            )

        # Get the result tensor (variable named 'result' or 'output')
        user_result = user_namespace.get("result")
        if user_result is None:
            user_result = user_namespace.get("output")

        expected_result = expected_namespace.get("result")
        if expected_result is None:
            expected_result = expected_namespace.get("output")

        if user_result is None:
            return CheckResult(
                is_correct=False,
                message="Variable 'result' or 'output' not found. Please save your result to this variable.",
                error_type="execution",
            )

        if expected_result is None:
            return CheckResult(
                is_correct=False,
                message="Expected solution does not define 'result' or 'output'",
                error_type="execution",
            )

        # Check if both are tensors
        if not isinstance(user_result, torch.Tensor):
            return CheckResult(
                is_correct=False,
                message=f"Result must be a tensor (got: {type(user_result).__name__})",
                error_type="execution",
            )

        if not isinstance(expected_result, torch.Tensor):
            return CheckResult(
                is_correct=False,
                message=f"Expected result should be a tensor (got: {type(expected_result).__name__})",
                error_type="execution",
            )

        # Check shape
        if user_result.shape != expected_result.shape:
            return CheckResult(
                is_correct=False,
                message=f"形状が違います - 期待: {tuple(expected_result.shape)}, 結果: {tuple(user_result.shape)}",
                error_type="shape",
                expected_shape=tuple(expected_result.shape),
                actual_shape=tuple(user_result.shape),
                execution_output=stdout,
                actual_values=self._format_tensor_preview(user_result),
                expected_values=self._format_tensor_preview(expected_result),
            )

        # Check values
        try:
            # Convert to same dtype for comparison
            user_float = user_result.float() if user_result.dtype != torch.float32 else user_result
            expected_float = expected_result.float() if expected_result.dtype != torch.float32 else expected_result

            values_match = torch.allclose(user_float, expected_float, rtol=self.rtol, atol=self.atol)
        except Exception as e:
            return CheckResult(
                is_correct=False,
                message=f"Value Comparison Error: {str(e)}",
                error_type="value",
                execution_output=stdout,
            )

        if not values_match:
            # Calculate some statistics for debugging
            diff = (user_result.float() - expected_result.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            return CheckResult(
                is_correct=False,
                message=(
                    f"形状は正しいですが、値が違います\n"
                    f"最大差分: {max_diff:.6e}\n"
                    f"平均差分: {mean_diff:.6e}"
                ),
                error_type="value",
                expected_shape=tuple(expected_result.shape),
                actual_shape=tuple(user_result.shape),
                execution_output=stdout,
                actual_values=self._format_tensor_preview(user_result),
                expected_values=self._format_tensor_preview(expected_result),
            )

        # All checks passed!
        return CheckResult(
            is_correct=True,
            message="✅ Correct! Shape and values match.",
            expected_shape=tuple(expected_result.shape),
            actual_shape=tuple(user_result.shape),
            execution_output=stdout,
            actual_values=self._format_tensor_preview(user_result),
            expected_values=self._format_tensor_preview(expected_result),
        )

    def _format_tensor_preview(self, tensor: torch.Tensor, max_elements: int = 50) -> str:
        """Format tensor for display, truncating if too large."""
        total_elements = tensor.numel()
        shape_str = f"shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}"

        if total_elements <= max_elements:
            # Show entire tensor
            with torch.no_grad():
                tensor_str = str(tensor.detach().cpu())
            return f"{shape_str}\n{tensor_str}"
        else:
            # Show truncated preview
            flat = tensor.flatten()[:max_elements]
            with torch.no_grad():
                values_str = str(flat.detach().cpu().tolist())
            return f"{shape_str}\nFirst {max_elements} values: {values_str}..."

    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code and return the namespace."""
        namespace = {"torch": torch, "F": torch.nn.functional}
        exec(code, namespace)
        return namespace

    def _execute_code_with_output(self, code: str) -> tuple[Dict[str, Any], str]:
        """Execute code and return the namespace and captured stdout."""
        namespace = {"torch": torch, "F": torch.nn.functional}

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            exec(code, namespace)
            stdout = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        return namespace, stdout

