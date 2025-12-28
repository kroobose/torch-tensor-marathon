"""Einstein Summation problems - advanced tensor operations with einsum."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_einsum_problems() -> List[Problem]:
    """Get all Einstein Summation category problems."""

    problems = [
        # Beginner level
        Problem(
            id="einsum_001",
            category="einsum",
            difficulty="beginner",
            title_ja="ベクトルの内積",
            title_en="Vector Dot Product",
            description_ja="2つの形状 [100] のベクトルの内積を計算してください（結果はスカラー）。",
            description_en="Compute the dot product of two vectors of shape [100] (result is a scalar).",
            hint_ja="torch.einsum('i,i->', a, b) を使用します。",
            hint_en="Use torch.einsum('i,i->', a, b).",
            setup_code="""a = torch.randn(100)
b = torch.randn(100)""",
            solution_code="result = torch.einsum('i,i->', a, b)",
            tags=["einsum", "dot_product", "vector"],
        ),

        Problem(
            id="einsum_002",
            category="einsum",
            difficulty="beginner",
            title_ja="行列とベクトルの積",
            title_en="Matrix-Vector Product",
            description_ja="形状 [10, 20] の行列と形状 [20] のベクトルの積を計算してください。結果の形状は [10] です。",
            description_en="Compute the product of a matrix of shape [10, 20] and a vector of shape [20]. Result shape is [10].",
            hint_ja="torch.einsum('ij,j->i', matrix, vector) を使用します。",
            hint_en="Use torch.einsum('ij,j->i', matrix, vector).",
            setup_code="""matrix = torch.randn(10, 20)
vector = torch.randn(20)""",
            solution_code="result = torch.einsum('ij,j->i', matrix, vector)",
            tags=["einsum", "matrix_vector", "linear_algebra"],
        ),

        Problem(
            id="einsum_003",
            category="einsum",
            difficulty="beginner",
            title_ja="行列の積",
            title_en="Matrix Multiplication",
            description_ja="形状 [10, 20] と [20, 30] の2つの行列の積を計算してください。結果の形状は [10, 30] です。",
            description_en="Compute the product of two matrices of shapes [10, 20] and [20, 30]. Result shape is [10, 30].",
            hint_ja="torch.einsum('ik,kj->ij', A, B) を使用します。",
            hint_en="Use torch.einsum('ik,kj->ij', A, B).",
            setup_code="""A = torch.randn(10, 20)
B = torch.randn(20, 30)""",
            solution_code="result = torch.einsum('ik,kj->ij', A, B)",
            tags=["einsum", "matmul", "linear_algebra"],
        ),

        Problem(
            id="einsum_004",
            category="einsum",
            difficulty="beginner",
            title_ja="転置",
            title_en="Transpose",
            description_ja="形状 [10, 20] の行列を転置してください。",
            description_en="Transpose a matrix of shape [10, 20].",
            hint_ja="torch.einsum('ij->ji', matrix) を使用します。",
            hint_en="Use torch.einsum('ij->ji', matrix).",
            setup_code="matrix = torch.randn(10, 20)",
            solution_code="result = torch.einsum('ij->ji', matrix)",
            tags=["einsum", "transpose"],
        ),

        # Intermediate level
        Problem(
            id="einsum_005",
            category="einsum",
            difficulty="intermediate",
            title_ja="バッチ行列積",
            title_en="Batch Matrix Multiplication",
            description_ja="形状 [32, 10, 20] と [32, 20, 30] のバッチ行列の積を計算してください。結果の形状は [32, 10, 30] です。",
            description_en="Compute batch matrix product of tensors with shapes [32, 10, 20] and [32, 20, 30]. Result shape is [32, 10, 30].",
            hint_ja="torch.einsum('bik,bkj->bij', A, B) を使用します。",
            hint_en="Use torch.einsum('bik,bkj->bij', A, B).",
            setup_code="""A = torch.randn(32, 10, 20)
B = torch.randn(32, 20, 30)""",
            solution_code="result = torch.einsum('bik,bkj->bij', A, B)",
            tags=["einsum", "batch", "matmul"],
        ),

        Problem(
            id="einsum_006",
            category="einsum",
            difficulty="intermediate",
            title_ja="対角要素の抽出",
            title_en="Extract Diagonal",
            description_ja="形状 [10, 10] の行列から対角要素を抽出してください。結果の形状は [10] です。",
            description_en="Extract the diagonal elements from a matrix of shape [10, 10]. Result shape is [10].",
            hint_ja="torch.einsum('ii->i', matrix) を使用します。",
            hint_en="Use torch.einsum('ii->i', matrix).",
            setup_code="matrix = torch.randn(10, 10)",
            solution_code="result = torch.einsum('ii->i', matrix)",
            tags=["einsum", "diagonal"],
        ),

        Problem(
            id="einsum_007",
            category="einsum",
            difficulty="intermediate",
            title_ja="トレースの計算",
            title_en="Compute Trace",
            description_ja="形状 [10, 10] の行列のトレース（対角要素の和）を計算してください（結果はスカラー）。",
            description_en="Compute the trace (sum of diagonal elements) of a matrix of shape [10, 10] (result is a scalar).",
            hint_ja="torch.einsum('ii->', matrix) を使用します。",
            hint_en="Use torch.einsum('ii->', matrix).",
            setup_code="matrix = torch.randn(10, 10)",
            solution_code="result = torch.einsum('ii->', matrix)",
            tags=["einsum", "trace", "linear_algebra"],
        ),

        Problem(
            id="einsum_008",
            category="einsum",
            difficulty="intermediate",
            title_ja="外積",
            title_en="Outer Product",
            description_ja="形状 [10] と [20] の2つのベクトルの外積を計算してください。結果の形状は [10, 20] です。",
            description_en="Compute the outer product of two vectors of shapes [10] and [20]. Result shape is [10, 20].",
            hint_ja="torch.einsum('i,j->ij', a, b) を使用します。",
            hint_en="Use torch.einsum('i,j->ij', a, b).",
            setup_code="""a = torch.randn(10)
b = torch.randn(20)""",
            solution_code="result = torch.einsum('i,j->ij', a, b)",
            tags=["einsum", "outer_product"],
        ),

        Problem(
            id="einsum_009",
            category="einsum",
            difficulty="intermediate",
            title_ja="要素ごとの積と和",
            title_en="Element-wise Product and Sum",
            description_ja="形状 [10, 20] の2つの行列の要素ごとの積を計算し、全要素を合計してください（結果はスカラー）。",
            description_en="Compute element-wise product of two matrices of shape [10, 20] and sum all elements (result is a scalar).",
            hint_ja="torch.einsum('ij,ij->', A, B) を使用します。",
            hint_en="Use torch.einsum('ij,ij->', A, B).",
            setup_code="""A = torch.randn(10, 20)
B = torch.randn(10, 20)""",
            solution_code="result = torch.einsum('ij,ij->', A, B)",
            tags=["einsum", "element_wise", "sum"],
        ),

        # Advanced level
        Problem(
            id="einsum_010",
            category="einsum",
            difficulty="advanced",
            title_ja="Attention スコアの計算 (Q @ K^T)",
            title_en="Attention Score Computation (Q @ K^T)",
            description_ja="形状 [32, 8, 128, 64] のクエリ Q と キー K の Attention スコアを計算してください。結果の形状は [32, 8, 128, 128] です。",
            description_en="Compute attention scores from queries Q and keys K of shape [32, 8, 128, 64]. Result shape is [32, 8, 128, 128].",
            hint_ja="torch.einsum('bhqd,bhkd->bhqk', Q, K) を使用します。",
            hint_en="Use torch.einsum('bhqd,bhkd->bhqk', Q, K).",
            setup_code="""Q = torch.randn(32, 8, 128, 64)
K = torch.randn(32, 8, 128, 64)""",
            solution_code="result = torch.einsum('bhqd,bhkd->bhqk', Q, K)",
            tags=["einsum", "attention", "nlp"],
        ),

        Problem(
            id="einsum_011",
            category="einsum",
            difficulty="advanced",
            title_ja="Attention の出力計算",
            title_en="Attention Output Computation",
            description_ja="形状 [32, 8, 128, 128] の Attention 重み と形状 [32, 8, 128, 64] の Value を使って出力を計算してください。結果の形状は [32, 8, 128, 64] です。",
            description_en="Compute attention output from attention weights of shape [32, 8, 128, 128] and values of shape [32, 8, 128, 64]. Result shape is [32, 8, 128, 64].",
            hint_ja="torch.einsum('bhqk,bhkd->bhqd', attn, V) を使用します。",
            hint_en="Use torch.einsum('bhqk,bhkd->bhqd', attn, V).",
            setup_code="""attn = torch.randn(32, 8, 128, 128)
V = torch.randn(32, 8, 128, 64)""",
            solution_code="result = torch.einsum('bhqk,bhkd->bhqd', attn, V)",
            tags=["einsum", "attention", "nlp"],
        ),

        Problem(
            id="einsum_012",
            category="einsum",
            difficulty="expert",
            title_ja="テンソル縮約: 4D → 2D",
            title_en="Tensor Contraction: 4D → 2D",
            description_ja="形状 [10, 20, 30, 40] のテンソル A と形状 [30, 40, 50, 60] のテンソル B を縮約して、形状 [10, 20, 50, 60] の結果を得てください。",
            description_en="Contract tensors A of shape [10, 20, 30, 40] and B of shape [30, 40, 50, 60] to get a result of shape [10, 20, 50, 60].",
            hint_ja="torch.einsum('ijkl,klmn->ijmn', A, B) を使用します。",
            hint_en="Use torch.einsum('ijkl,klmn->ijmn', A, B).",
            setup_code="""A = torch.randn(10, 20, 30, 40)
B = torch.randn(30, 40, 50, 60)""",
            solution_code="result = torch.einsum('ijkl,klmn->ijmn', A, B)",
            tags=["einsum", "contraction", "4d"],
        ),
    ]

    return problems
