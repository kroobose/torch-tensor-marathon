"""Correctness checker for validating user solutions."""

from dataclasses import dataclass
from typing import Dict, Any
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
                message=f"Shape Error: Expected {tuple(expected_result.shape)}, Got {tuple(user_result.shape)}",
                error_type="shape",
                expected_shape=tuple(expected_result.shape),
                actual_shape=tuple(user_result.shape),
                execution_output=stdout,
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
                    f"Value Error: Tensor values do not match\n"
                    f"Max Diff: {max_diff:.6e}\n"
                    f"Mean Diff: {mean_diff:.6e}\n"
                    f"Tolerance: rtol={self.rtol}, atol={self.atol}"
                ),
                error_type="value",
                expected_shape=tuple(expected_result.shape),
                actual_shape=tuple(user_result.shape),
                execution_output=stdout,
            )

        # All checks passed!
        return CheckResult(
            is_correct=True,
            message="✅ Correct! Shape and values match.",
            expected_shape=tuple(expected_result.shape),
            actual_shape=tuple(user_result.shape),
            execution_output=stdout,
        )

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
