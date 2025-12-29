"""Einsum problems - explicit tensor contraction."""

from typing import List
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_einsum_problems() -> List[Problem]:
    """Get all Einsum category problems."""

    problems = [
        Problem(
            id="einsum_reductions",
            category="einsum",
            difficulty="intermediate",
            title_ja="Einsum Reductions",
            title_en="Einsum Reductions",
            cases=[
                ProblemCase(
                    name="Dot Product",
                    description_ja="ベクトル x [5] と y [5] の内積を einsum で計算してください。",
                    description_en="Compute dot product of x and y using einsum.",
                    hint_ja="torch.einsum('i,i->', x, y) を使用します。",
                    hint_en="Use torch.einsum('i,i->', x, y).",
                    setup_code="""x = torch.randn(5)
y = torch.randn(5)""",
                    solution_code="result = torch.einsum('i,i->', x, y)"
                ),
                ProblemCase(
                    name="Total Sum",
                    description_ja="テンソル x [3, 4] の全要素の和を einsum で計算してください。",
                    description_en="Compute sum of all elements in x using einsum.",
                    hint_ja="torch.einsum('ij->', x) を使用します。",
                    hint_en="Use torch.einsum('ij->', x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.einsum('ij->', x)"
                ),
                ProblemCase(
                    name="Row Sum",
                    description_ja="テンソル x [3, 4] の行ごとの和 [3] (dim=1の和) を einsum で計算してください。",
                    description_en="Compute row sums (sum over dim 1) using einsum.",
                    hint_ja="torch.einsum('ij->i', x) を使用します。",
                    hint_en="Use torch.einsum('ij->i', x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.einsum('ij->i', x)"
                ),
                ProblemCase(
                    name="Trace",
                    description_ja="正方行列 x [4, 4] のトレース（対角成分の和）を einsum で計算してください。",
                    description_en="Compute trace of x using einsum.",
                    hint_ja="torch.einsum('ii->', x) を使用します。",
                    hint_en="Use torch.einsum('ii->', x).",
                    setup_code="x = torch.randn(4, 4)",
                    solution_code="result = torch.einsum('ii->', x)"
                ),
                 ProblemCase(
                    name="Sum of Squares",
                    description_ja="テンソル x の全要素の二乗和を einsum で計算してください。",
                    description_en="Compute sum of squares of x using einsum.",
                    hint_ja="torch.einsum('ij,ij->', x, x) を使用します。",
                    hint_en="Use torch.einsum('ij,ij->', x, x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.einsum('ij,ij->', x, x)"
                ),
            ],
            tags=["einsum", "reduction"],
        ),

        Problem(
            id="einsum_structural",
            category="einsum",
            difficulty="intermediate",
            title_ja="Einsum Structural Ops",
            title_en="Einsum Structural Ops",
            cases=[
                ProblemCase(
                    name="Transpose",
                    description_ja="テンソル x [3, 4] を einsum で転置してください。",
                    description_en="Transpose x using einsum.",
                    hint_ja="torch.einsum('ij->ji', x) を使用します。",
                    hint_en="Use torch.einsum('ij->ji', x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.einsum('ij->ji', x)"
                ),
                ProblemCase(
                    name="Outer Product",
                    description_ja="ベクトル x [5] と y [4] の外積 [5, 4] を einsum で計算してください。",
                    description_en="Compute outer product of x and y using einsum.",
                    hint_ja="torch.einsum('i,j->ij', x, y) を使用します。",
                    hint_en="Use torch.einsum('i,j->ij', x, y).",
                    setup_code="""x = torch.randn(5)
y = torch.randn(4)""",
                    solution_code="result = torch.einsum('i,j->ij', x, y)"
                ),
                ProblemCase(
                    name="Diagonal Extraction",
                    description_ja="正方行列 x [4, 4] の対角成分を einsum で抽出してください。",
                    description_en="Extract diagonal of x using einsum.",
                    hint_ja="torch.einsum('ii->i', x) を使用します。",
                    hint_en="Use torch.einsum('ii->i', x).",
                    setup_code="x = torch.randn(4, 4)",
                    solution_code="result = torch.einsum('ii->i', x)"
                ),
                ProblemCase(
                    name="Hadamard Product",
                    description_ja="同じ形状の3つのテンソル A, B, C の要素ごとの積を einsum で計算してください。",
                    description_en="Compute Hadamard product of A, B, C using einsum.",
                    hint_ja="torch.einsum('ijk,ijk,ijk->ijk', A, B, C) を使用します。",
                    hint_en="Use torch.einsum('ijk,ijk,ijk->ijk', A, B, C).",
                    setup_code="""A = torch.randn(3, 4, 5)
B = torch.randn(3, 4, 5)
C = torch.randn(3, 4, 5)""",
                    solution_code="result = torch.einsum('ijk,ijk,ijk->ijk', A, B, C)"
                ),
            ],
            tags=["einsum", "transpose", "outer", "diagonal"],
        ),

        Problem(
            id="einsum_matmul_family",
            category="einsum",
            difficulty="advanced",
            title_ja="Einsum Matmul & Apps",
            title_en="Einsum Matmul & Apps",
            cases=[
                ProblemCase(
                    name="Matrix Multiplication",
                    description_ja="行列 A [3, 4] と B [4, 5] の積 [3, 5] を einsum で計算してください。",
                    description_en="Compute A @ B using einsum.",
                    hint_ja="torch.einsum('ij,jk->ik', A, B) を使用します。",
                    hint_en="Use torch.einsum('ij,jk->ik', A, B).",
                    setup_code="""A = torch.randn(3, 4)
B = torch.randn(4, 5)""",
                    solution_code="result = torch.einsum('ij,jk->ik', A, B)"
                ),
                ProblemCase(
                    name="Batch Matmul",
                    description_ja="バッチ行列積 A [B, I, K] と C [B, K, J] -> [B, I, J] を einsum で計算してください。",
                    description_en="Compute batch matmul A @ C using einsum.",
                    hint_ja="torch.einsum('bik,bkj->bij', A, C) を使用します。",
                    hint_en="Use torch.einsum('bik,bkj->bij', A, C).",
                    setup_code="""B, I, J, K = 2, 3, 4, 5
A = torch.randn(B, I, K)
C = torch.randn(B, K, J)""",
                    solution_code="result = torch.einsum('bik,bkj->bij', A, C)"
                ),
                ProblemCase(
                    name="Attention Logits",
                    description_ja="Q [B, H, L, D] と K [B, H, S, D] からAttention Logits [B, H, L, S] を計算してください。",
                    description_en="Compute Attention Logits Q @ K^T.",
                    hint_ja="torch.einsum('bhld,bhsd->bhls', Q, K) を使用します。",
                    hint_en="Use torch.einsum('bhld,bhsd->bhls', Q, K).",
                    setup_code="""B, H, L, S, D = 2, 4, 8, 8, 16
Q = torch.randn(B, H, L, D)
K = torch.randn(B, H, S, D)""",
                    solution_code="result = torch.einsum('bhld,bhsd->bhls', Q, K)"
                ),
            ],
            tags=["einsum", "matmul", "attention"],
        ),

        Problem(
            id="einsum_multilinear",
            category="einsum",
            difficulty="expert",
            title_ja="Advanced Einsum (Multilinear)",
            title_en="Advanced Einsum (Multilinear)",
            cases=[
                ProblemCase(
                    name="Bilinear Form",
                    description_ja="双一次形式 x [10, I], W [K, I, J], y [10, J] -> [10, K] を計算してください ('bi,kij,bj->bk')。",
                    description_en="Compute bilinear form.",
                    hint_ja="torch.einsum('bi,kij,bj->bk', x, W, y) を使用します。",
                    hint_en="Use torch.einsum('bi,kij,bj->bk', x, W, y).",
                    setup_code="""I, J, K = 3, 4, 5
x = torch.randn(10, I)
y = torch.randn(10, J)
W = torch.randn(K, I, J)""",
                    solution_code="result = torch.einsum('bi,kij,bj->bk', x, W, y)"
                ),
            ],
            tags=["einsum", "bilinear"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="einsum_attention_full",
            category="einsum",
            difficulty="expert",
            title_ja="Einsum Attention Computation",
            title_en="Einsum Attention Computation",
            cases=[
                ProblemCase(
                    name="Attention Output",
                    description_ja="Attention出力: attn_weights [B, H, L, S] と V [B, H, S, D] から出力 [B, H, L, D] を計算してください。",
                    description_en="Compute attention output: attn_weights @ V.",
                    hint_ja="torch.einsum('bhls,bhsd->bhld', attn_weights, V) を使用します。",
                    hint_en="Use torch.einsum('bhls,bhsd->bhld', attn_weights, V).",
                    setup_code="""B, H, L, S, D = 2, 4, 8, 8, 16
attn_weights = torch.softmax(torch.randn(B, H, L, S), dim=-1)
V = torch.randn(B, H, S, D)""",
                    solution_code="result = torch.einsum('bhls,bhsd->bhld', attn_weights, V)"
                ),
                ProblemCase(
                    name="Full Attention",
                    description_ja="Q, K, V から完全なAttention出力を計算してください (softmax込み)。",
                    description_en="Compute full attention: softmax(QK^T/sqrt(d)) @ V.",
                    hint_ja="einsum でロジットを計算し、softmax後に再度einsum。",
                    hint_en="Compute logits with einsum, softmax, then einsum with V.",
                    setup_code="""B, H, L, D = 2, 4, 8, 16
Q = torch.randn(B, H, L, D)
K = torch.randn(B, H, L, D)
V = torch.randn(B, H, L, D)
scale = D ** 0.5""",
                    solution_code="""logits = torch.einsum('bhld,bhsd->bhls', Q, K) / scale
attn = torch.softmax(logits, dim=-1)
result = torch.einsum('bhls,bhsd->bhld', attn, V)"""
                ),
            ],
            tags=["einsum", "attention"],
        ),

        Problem(
            id="einsum_elementwise",
            category="einsum",
            difficulty="beginner",
            title_ja="Einsum Element-wise Operations",
            title_en="Einsum Element-wise Operations",
            cases=[
                ProblemCase(
                    name="Element-wise Mul",
                    description_ja="テンソル x と y [3, 4] の要素ごとの積を einsum で計算してください。",
                    description_en="Compute element-wise product using einsum.",
                    hint_ja="torch.einsum('ij,ij->ij', x, y) を使用します。",
                    hint_en="Use torch.einsum('ij,ij->ij', x, y).",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(3, 4)""",
                    solution_code="result = torch.einsum('ij,ij->ij', x, y)"
                ),
                ProblemCase(
                    name="Weighted Sum",
                    description_ja="重み w [3] とテンソル x [3, 4] の加重和 [4] を einsum で計算してください。",
                    description_en="Compute weighted sum over first dim.",
                    hint_ja="torch.einsum('i,ij->j', w, x) を使用します。",
                    hint_en="Use torch.einsum('i,ij->j', w, x).",
                    setup_code="""w = torch.softmax(torch.randn(3), dim=0)
x = torch.randn(3, 4)""",
                    solution_code="result = torch.einsum('i,ij->j', w, x)"
                ),
            ],
            tags=["einsum", "elementwise"],
        ),

        Problem(
            id="einsum_kronecker",
            category="einsum",
            difficulty="advanced",
            title_ja="Einsum Kronecker Product",
            title_en="Einsum Kronecker Product",
            cases=[
                ProblemCase(
                    name="Kronecker 2x2",
                    description_ja="行列 A [2, 2] と B [3, 3] のクロネッカー積 [6, 6] を einsum とreshapeで計算してください。",
                    description_en="Compute Kronecker product of A and B.",
                    hint_ja="torch.einsum('ij,kl->ikjl', A, B).reshape(6, 6) を使用します。",
                    hint_en="Use torch.einsum('ij,kl->ikjl', A, B).reshape(6, 6).",
                    setup_code="""A = torch.randn(2, 2)
B = torch.randn(3, 3)""",
                    solution_code="result = torch.einsum('ij,kl->ikjl', A, B).reshape(6, 6)"
                ),
            ],
            tags=["einsum", "kronecker"],
        ),

        Problem(
            id="einsum_tensor_contraction",
            category="einsum",
            difficulty="expert",
            title_ja="Einsum Tensor Contraction",
            title_en="Einsum Tensor Contraction",
            cases=[
                ProblemCase(
                    name="3-Tensor Contraction",
                    description_ja="テンソル A [I, J, K], B [K, L], C [L, M] を縮約して [I, J, M] にしてください。",
                    description_en="Contract A, B, C to get [I, J, M].",
                    hint_ja="torch.einsum('ijk,kl,lm->ijm', A, B, C) を使用します。",
                    hint_en="Use torch.einsum('ijk,kl,lm->ijm', A, B, C).",
                    setup_code="""I, J, K, L, M = 2, 3, 4, 5, 6
A = torch.randn(I, J, K)
B = torch.randn(K, L)
C = torch.randn(L, M)""",
                    solution_code="result = torch.einsum('ijk,kl,lm->ijm', A, B, C)"
                ),
                ProblemCase(
                    name="Double Contraction",
                    description_ja="テンソル A [I, J, K] と B [K, J, L] を (J, K) で縮約して [I, L] にしてください。",
                    description_en="Contract A and B over two indices.",
                    hint_ja="torch.einsum('ijk,kjl->il', A, B) を使用します。",
                    hint_en="Use torch.einsum('ijk,kjl->il', A, B).",
                    setup_code="""I, J, K, L = 2, 3, 4, 5
A = torch.randn(I, J, K)
B = torch.randn(K, J, L)""",
                    solution_code="result = torch.einsum('ijk,kjl->il', A, B)"
                ),
            ],
            tags=["einsum", "contraction"],
        ),

        Problem(
            id="einsum_gram_matrix",
            category="einsum",
            difficulty="intermediate",
            title_ja="Einsum Gram Matrix",
            title_en="Einsum Gram Matrix",
            cases=[
                ProblemCase(
                    name="Gram Matrix",
                    description_ja="特徴行列 X [N, D] からグラム行列 G [N, N] (G = X @ X^T) を計算してください。",
                    description_en="Compute Gram matrix G = X @ X^T.",
                    hint_ja="torch.einsum('ij,kj->ik', X, X) を使用します。",
                    hint_en="Use torch.einsum('ij,kj->ik', X, X).",
                    setup_code="X = torch.randn(10, 5)",
                    solution_code="result = torch.einsum('ij,kj->ik', X, X)"
                ),
                ProblemCase(
                    name="Covariance Style",
                    description_ja="特徴行列 X [N, D] と Y [N, D] から X^T @ Y を計算してください。",
                    description_en="Compute X^T @ Y.",
                    hint_ja="torch.einsum('ni,nj->ij', X, Y) を使用します。",
                    hint_en="Use torch.einsum('ni,nj->ij', X, Y).",
                    setup_code="""X = torch.randn(10, 5)
Y = torch.randn(10, 6)""",
                    solution_code="result = torch.einsum('ni,nj->ij', X, Y)"
                ),
            ],
            tags=["einsum", "gram"],
        ),

        Problem(
            id="einsum_batch_outer",
            category="einsum",
            difficulty="intermediate",
            title_ja="Einsum Batched Outer Product",
            title_en="Einsum Batched Outer Product",
            cases=[
                ProblemCase(
                    name="Batch Outer",
                    description_ja="バッチベクトル x [B, I] と y [B, J] のバッチ外積 [B, I, J] を計算してください。",
                    description_en="Compute batched outer product.",
                    hint_ja="torch.einsum('bi,bj->bij', x, y) を使用します。",
                    hint_en="Use torch.einsum('bi,bj->bij', x, y).",
                    setup_code="""B, I, J = 4, 5, 6
x = torch.randn(B, I)
y = torch.randn(B, J)""",
                    solution_code="result = torch.einsum('bi,bj->bij', x, y)"
                ),
                ProblemCase(
                    name="Triple Batch Sum",
                    description_ja="バッチテンソル A, B, C [B, N] の要素ごとの積の合計 [B] を計算してください。",
                    description_en="Compute sum of element-wise product A * B * C per batch.",
                    hint_ja="torch.einsum('bi,bi,bi->b', A, B, C) を使用します。",
                    hint_en="Use torch.einsum('bi,bi,bi->b', A, B, C).",
                    setup_code="""B, N = 4, 10
A = torch.randn(B, N)
B_ = torch.randn(B, N)
C = torch.randn(B, N)""",
                    solution_code="result = torch.einsum('bi,bi,bi->b', A, B_, C)"
                ),
            ],
            tags=["einsum", "batch", "outer"],
        ),

        Problem(
            id="einsum_complex_index",
            category="einsum",
            difficulty="advanced",
            title_ja="Einsum Complex Index Patterns",
            title_en="Einsum Complex Index Patterns",
            cases=[
                ProblemCase(
                    name="Permute 4D",
                    description_ja="4Dテンソル x [A, B, C, D] を einsum で [B, D, A, C] に並び替えてください。",
                    description_en="Permute x to [B, D, A, C] using einsum.",
                    hint_ja="torch.einsum('abcd->bdac', x) を使用します。",
                    hint_en="Use torch.einsum('abcd->bdac', x).",
                    setup_code="x = torch.randn(2, 3, 4, 5)",
                    solution_code="result = torch.einsum('abcd->bdac', x)"
                ),
                ProblemCase(
                    name="Selective Sum",
                    description_ja="テンソル x [A, B, C] の dim=1 のみを縮約して [A, C] にしてください。",
                    description_en="Sum over dim 1 only using einsum.",
                    hint_ja="torch.einsum('abc->ac', x) を使用します。",
                    hint_en="Use torch.einsum('abc->ac', x).",
                    setup_code="x = torch.randn(2, 3, 4)",
                    solution_code="result = torch.einsum('abc->ac', x)"
                ),
            ],
            tags=["einsum", "permute", "sum"],
        ),

        Problem(
            id="einsum_vs_native",
            category="einsum",
            difficulty="beginner",
            title_ja="Einsum vs Native Ops",
            title_en="Einsum vs Native Ops",
            cases=[
                ProblemCase(
                    name="Sum via Einsum",
                    description_ja="torch.sum(x, dim=1) と同等の操作を einsum で書いてください。",
                    description_en="Write torch.sum(x, dim=1) using einsum.",
                    hint_ja="torch.einsum('ij->i', x) を使用します。",
                    hint_en="Use torch.einsum('ij->i', x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.einsum('ij->i', x)"
                ),
                ProblemCase(
                    name="Matmul via Einsum",
                    description_ja="torch.matmul(A, B) と同等の操作を einsum で書いてください。",
                    description_en="Write torch.matmul(A, B) using einsum.",
                    hint_ja="torch.einsum('ij,jk->ik', A, B) を使用します。",
                    hint_en="Use torch.einsum('ij,jk->ik', A, B).",
                    setup_code="""A = torch.randn(3, 4)
B = torch.randn(4, 5)""",
                    solution_code="result = torch.einsum('ij,jk->ik', A, B)"
                ),
            ],
            tags=["einsum", "comparison"],
        ),
    ]

    return problems

