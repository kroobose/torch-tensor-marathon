"""Problem bank initialization - imports all problem categories."""

from torch_tensor_marathon.problems.reshape_permute import get_reshape_permute_problems
from torch_tensor_marathon.problems.indexing_slicing import get_indexing_slicing_problems
from torch_tensor_marathon.problems.broadcasting import get_broadcasting_problems
from torch_tensor_marathon.problems.gather_scatter import get_gather_scatter_problems
from torch_tensor_marathon.problems.einsum import get_einsum_problems
from torch_tensor_marathon.problems.stacking_splitting import get_stacking_splitting_problems
from torch_tensor_marathon.problems.advanced_ops import get_advanced_ops_problems
from torch_tensor_marathon.problems.dl_applications import get_dl_applications_problems

from torch_tensor_marathon.problem import problem_bank


def initialize_problems():
    """Load all problems into the global problem bank."""
    problem_bank.add_problems(get_reshape_permute_problems())
    problem_bank.add_problems(get_indexing_slicing_problems())
    problem_bank.add_problems(get_broadcasting_problems())
    problem_bank.add_problems(get_gather_scatter_problems())
    problem_bank.add_problems(get_einsum_problems())
    problem_bank.add_problems(get_stacking_splitting_problems())
    problem_bank.add_problems(get_advanced_ops_problems())
    problem_bank.add_problems(get_dl_applications_problems())

    return problem_bank


__all__ = ["initialize_problems"]
