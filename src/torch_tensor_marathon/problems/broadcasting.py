"""Broadcasting problems - implicit shape expansion."""

from typing import List
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_broadcasting_problems() -> List[Problem]:
    """Get all Broadcasting category problems."""

    problems = [
        Problem(
            id="broadcast_scalar",
            category="broadcasting",
            difficulty="beginner",
            title_ja="Scalar Broadcasting",
            title_en="Scalar Broadcasting",
            cases=[
                ProblemCase(
                    name="Add Scalar",
                    description_ja="テンソル x [3] にスカラー 5.0 を足してください。",
                    description_en="Add scalar 5.0 to tensor x [3].",
                    hint_ja="x + 5.0 を使用します。",
                    hint_en="Use x + 5.0.",
                    setup_code="x = torch.zeros(3)",
                    solution_code="result = x + 5.0"
                ),
                ProblemCase(
                    name="Multiply Scalar",
                    description_ja="テンソル x [2, 2] にスカラー 3.0 を掛けてください。",
                    description_en="Multiply tensor x [2, 2] by scalar 3.0.",
                    hint_ja="x * 3.0 を使用します。",
                    hint_en="Use x * 3.0.",
                    setup_code="x = torch.ones(2, 2)",
                    solution_code="result = x * 3.0"
                ),
            ],
            tags=["broadcasting", "arithmetic"],
        ),

        Problem(
            id="broadcast_vector_2d",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="Vector Broadcasting",
            title_en="Vector Broadcasting",
            cases=[
                ProblemCase(
                    name="Add Row Vector",
                    description_ja="テンソル x [3, 4] にベクトル y [4] を足してください (各行に加算)。",
                    description_en="Add y [4] to x [3, 4] (add to each row).",
                    hint_ja="x + y を使用します。",
                    hint_en="Use x + y.",
                    setup_code="""x = torch.zeros(3, 4)
y = torch.arange(4)""",
                    solution_code="result = x + y"
                ),
                ProblemCase(
                    name="Add Col Vector",
                    description_ja="テンソル x [3, 4] に列ベクトル y [3, 1] を足してください (各列に加算)。",
                    description_en="Add y [3, 1] to x [3, 4] (add to each col).",
                    hint_ja="x + y を使用します。",
                    hint_en="Use x + y.",
                    setup_code="""x = torch.zeros(3, 4)
y = torch.arange(3).unsqueeze(1)""",
                    solution_code="result = x + y"
                ),
                ProblemCase(
                    name="Outer Product",
                    description_ja="ベクトル x [3] と y [4] をブロードキャストして、形状 [3, 4] の和を作ってください。",
                    description_en="Broadcast x [3] and y [4] to get sum of shape [3, 4].",
                    hint_ja="x.unsqueeze(1) + y.unsqueeze(0) を使用します。",
                    hint_en="Use x.unsqueeze(1) + y.unsqueeze(0).",
                    setup_code="""x = torch.arange(3)
y = torch.arange(4)""",
                    solution_code="result = x.unsqueeze(1) + y.unsqueeze(0)"
                ),
            ],
            tags=["broadcasting"],
        ),

        Problem(
            id="broadcast_3d",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="3D Broadcasting",
            title_en="3D Broadcasting",
            cases=[
                ProblemCase(
                    name="3D + 1D (Last)",
                    description_ja="テンソル x [2, 3, 4] にベクトル y [4] を足してください。",
                    description_en="Add y [4] to x [2, 3, 4].",
                    hint_ja="x + y を使用します。",
                    hint_en="Use x + y.",
                    setup_code="""x = torch.zeros(2, 3, 4)
y = torch.arange(4)""",
                    solution_code="result = x + y"
                ),
                ProblemCase(
                    name="3D + 2D",
                    description_ja="テンソル x [2, 3, 4] に y [3, 4] を足してください。",
                    description_en="Add y [3, 4] to x [2, 3, 4].",
                    hint_ja="x + y を使用します。",
                    hint_en="Use x + y.",
                    setup_code="""x = torch.zeros(2, 3, 4)
y = torch.ones(3, 4)""",
                    solution_code="result = x + y"
                ),
                ProblemCase(
                    name="3D + 1D (Middle)",
                    description_ja="テンソル x [2, 3, 4] にベクトル y [3] を dim=1 に沿って足してください。",
                    description_en="Add y [3] to x [2, 3, 4] along dim 1.",
                    hint_ja="x + y.view(1, 3, 1) を使用します。",
                    hint_en="Use x + y.view(1, 3, 1).",
                    setup_code="""x = torch.zeros(2, 3, 4)
y = torch.arange(3)""",
                    solution_code="result = x + y.view(1, 3, 1)"
                ),
            ],
            tags=["broadcasting"],
        ),

        Problem(
            id="broadcast_explicit",
            category="broadcasting",
            difficulty="advanced",
            title_ja="Explicit Broadcasting",
            title_en="Explicit Broadcasting",
            cases=[
                ProblemCase(
                    name="broadcast_to",
                    description_ja="テンソル x [3, 1] を [3, 4] にメモリコピーなしでブロードキャストしてください。",
                    description_en="Broadcast x [3, 1] to [3, 4] without copy.",
                    hint_ja="torch.broadcast_to(x, (3, 4)) を使用します。",
                    hint_en="Use torch.broadcast_to(x, (3, 4)).",
                    setup_code="x = torch.randn(3, 1)",
                    solution_code="result = torch.broadcast_to(x, (3, 4))"
                ),
                ProblemCase(
                    name="broadcast_tensors",
                    description_ja="テンソル x [3, 1] と y [1, 4] を共通の形状 [3, 4] にブロードキャストしてください。",
                    description_en="Broadcast x and y to common shape [3, 4].",
                    hint_ja="torch.broadcast_tensors(x, y) (戻り値はタプル)",
                    hint_en="Use torch.broadcast_tensors(x, y).",
                    setup_code="""x = torch.randn(3, 1)
y = torch.randn(1, 4)""",
                    solution_code="result = torch.broadcast_tensors(x, y)"
                ),
            ],
            tags=["broadcast_to", "broadcast_tensors"],
        ),

        Problem(
            id="broadcast_applications",
            category="broadcasting",
            difficulty="expert",
            title_ja="Broadcasting Applications",
            title_en="Broadcasting Applications",
            cases=[
                ProblemCase(
                    name="Batch Matmul",
                    description_ja="バッチ行列積: A [10, 3, 4] と B [4, 5] (broadcasted to [10, 4, 5]) の積を計算してください。",
                    description_en="Batch matmul: A [10, 3, 4] @ B [4, 5].",
                    hint_ja="torch.matmul(A, B) または A @ B を使用します。",
                    hint_en="Use A @ B.",
                    setup_code="""A = torch.randn(10, 3, 4)
B = torch.randn(4, 5)""",
                    solution_code="result = A @ B"
                ),
                 ProblemCase(
                    name="Pairwise Distance",
                    description_ja="点群 A [N, D] と B [M, D] の全ペア間の差 (N, M, D) を計算してください。",
                    description_en="Compute diff between all pairs in A and B.",
                    hint_ja="A.unsqueeze(1) - B.unsqueeze(0) を使用します。",
                    hint_en="Use A.unsqueeze(1) - B.unsqueeze(0).",
                    setup_code="""A = torch.randn(5, 2)
B = torch.randn(4, 2)""",
                    solution_code="result = A.unsqueeze(1) - B.unsqueeze(0)"
                ),
                ProblemCase(
                    name="Masking",
                    description_ja="画像バッチ [N, C, H, W] にマスク [1, 1, H, W] を適用（掛け算）してください。",
                    description_en="Apply mask [1, 1, H, W] to batch [N, C, H, W].",
                    hint_ja="images * mask を使用します。",
                    hint_en="Use images * mask.",
                    setup_code="""images = torch.randn(4, 3, 32, 32)
mask = torch.randint(0, 2, (1, 1, 32, 32)).float()""",
                    solution_code="result = images * mask"
                ),
            ],
            tags=["broadcasting", "applications"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="outer_product_ops",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="Outer Product Operations",
            title_en="Outer Product Operations",
            cases=[
                ProblemCase(
                    name="Vector Outer",
                    description_ja="ベクトル x [4] と y [5] の外積を計算して [4, 5] の行列を作ってください。",
                    description_en="Compute outer product of x [4] and y [5].",
                    hint_ja="torch.outer(x, y) を使用します。",
                    hint_en="Use torch.outer(x, y).",
                    setup_code="""x = torch.randn(4)
y = torch.randn(5)""",
                    solution_code="result = torch.outer(x, y)"
                ),
                ProblemCase(
                    name="Outer via Broadcast",
                    description_ja="ベクトル x [4] と y [5] の外積をブロードキャストで計算してください。",
                    description_en="Compute outer product using broadcasting.",
                    hint_ja="x.unsqueeze(1) * y.unsqueeze(0) を使用します。",
                    hint_en="Use x.unsqueeze(1) * y.unsqueeze(0).",
                    setup_code="""x = torch.randn(4)
y = torch.randn(5)""",
                    solution_code="result = x.unsqueeze(1) * y.unsqueeze(0)"
                ),
            ],
            tags=["outer", "broadcasting"],
        ),

        Problem(
            id="batch_matvec",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="Batch Matrix-Vector Multiply",
            title_en="Batch Matrix-Vector Multiply",
            cases=[
                ProblemCase(
                    name="Batch Matvec",
                    description_ja="バッチ行列 A [8, 3, 4] とベクトル v [4] の積を計算して [8, 3] にしてください。",
                    description_en="Compute A [8, 3, 4] @ v [4] to get [8, 3].",
                    hint_ja="(A @ v.unsqueeze(-1)).squeeze(-1) または torch.einsum('bij,j->bi', A, v) を使用します。",
                    hint_en="Use (A @ v.unsqueeze(-1)).squeeze(-1).",
                    setup_code="""A = torch.randn(8, 3, 4)
v = torch.randn(4)""",
                    solution_code="result = (A @ v.unsqueeze(-1)).squeeze(-1)"
                ),
                ProblemCase(
                    name="Vector Batch Mat",
                    description_ja="ベクトル v [3] とバッチ行列 A [8, 3, 4] の積を計算して [8, 4] にしてください。",
                    description_en="Compute v [3] @ A [8, 3, 4] to get [8, 4].",
                    hint_ja="(v.unsqueeze(0).unsqueeze(0) @ A).squeeze(-2) を使用します。",
                    hint_en="Use (v @ A) or einsum.",
                    setup_code="""v = torch.randn(3)
A = torch.randn(8, 3, 4)""",
                    solution_code="result = torch.einsum('j,ijk->ik', v, A)"
                ),
            ],
            tags=["batch", "matvec"],
        ),

        Problem(
            id="row_col_wise_ops",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="Row/Column-wise Operations",
            title_en="Row/Column-wise Operations",
            cases=[
                ProblemCase(
                    name="Row Mean Subtract",
                    description_ja="行列 x [4, 5] の各行から、その行の平均を引いてください。",
                    description_en="Subtract row mean from each row of x.",
                    hint_ja="x - x.mean(dim=1, keepdim=True) を使用します。",
                    hint_en="Use x - x.mean(dim=1, keepdim=True).",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = x - x.mean(dim=1, keepdim=True)"
                ),
                ProblemCase(
                    name="Col Normalize",
                    description_ja="行列 x [4, 5] の各列を、その列の最大値で正規化してください。",
                    description_en="Normalize each column by its max.",
                    hint_ja="x / x.max(dim=0, keepdim=True).values を使用します。",
                    hint_en="Use x / x.max(dim=0, keepdim=True).values.",
                    setup_code="x = torch.rand(4, 5) + 0.1",
                    solution_code="result = x / x.max(dim=0, keepdim=True).values"
                ),
                ProblemCase(
                    name="Row Sum Divide",
                    description_ja="行列 x [4, 5] の各行を、その行の合計で割ってください (確率分布化)。",
                    description_en="Divide each row by its sum (normalize to probability).",
                    hint_ja="x / x.sum(dim=1, keepdim=True) を使用します。",
                    hint_en="Use x / x.sum(dim=1, keepdim=True).",
                    setup_code="x = torch.rand(4, 5) + 0.1",
                    solution_code="result = x / x.sum(dim=1, keepdim=True)"
                ),
            ],
            tags=["row_wise", "col_wise"],
        ),

        Problem(
            id="expand_patterns",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="Expand Patterns",
            title_en="Expand Patterns",
            cases=[
                ProblemCase(
                    name="Expand As",
                    description_ja="テンソル x [1, 4] を target [3, 4] の形状に拡張してください。",
                    description_en="Expand x [1, 4] to match target [3, 4].",
                    hint_ja="x.expand_as(target) を使用します。",
                    hint_en="Use x.expand_as(target).",
                    setup_code="""x = torch.randn(1, 4)
target = torch.randn(3, 4)""",
                    solution_code="result = x.expand_as(target)"
                ),
                ProblemCase(
                    name="Expand -1",
                    description_ja="テンソル x [1, 3, 1] を [4, 3, 5] に拡張してください (-1 で元のサイズを維持)。",
                    description_en="Expand x [1, 3, 1] to [4, 3, 5].",
                    hint_ja="x.expand(4, -1, 5) を使用します。",
                    hint_en="Use x.expand(4, -1, 5).",
                    setup_code="x = torch.randn(1, 3, 1)",
                    solution_code="result = x.expand(4, -1, 5)"
                ),
            ],
            tags=["expand"],
        ),

        Problem(
            id="broadcasting_reduction",
            category="broadcasting",
            difficulty="advanced",
            title_ja="Broadcasting with Reduction",
            title_en="Broadcasting with Reduction",
            cases=[
                ProblemCase(
                    name="Euclidean Dist Row",
                    description_ja="行列 x [N, D] の各行から点 p [D] へのユークリッド距離を計算してください。",
                    description_en="Compute Euclidean distance from each row of x to p.",
                    hint_ja="((x - p) ** 2).sum(dim=1).sqrt() を使用します。",
                    hint_en="Use ((x - p) ** 2).sum(dim=1).sqrt().",
                    setup_code="""x = torch.randn(10, 3)
p = torch.randn(3)""",
                    solution_code="result = ((x - p) ** 2).sum(dim=1).sqrt()"
                ),
                ProblemCase(
                    name="Pairwise Cosine",
                    description_ja="行列 A [N, D] と B [M, D] の全ペア間のコサイン類似度を計算してください。",
                    description_en="Compute pairwise cosine similarity between A and B.",
                    hint_ja="F.normalize + matmul を使用します。",
                    hint_en="Use F.normalize then matmul.",
                    setup_code="""import torch.nn.functional as F
A = torch.randn(5, 8)
B = torch.randn(4, 8)""",
                    solution_code="result = F.normalize(A, dim=1) @ F.normalize(B, dim=1).T"
                ),
            ],
            tags=["broadcasting", "reduction"],
        ),

        Problem(
            id="implicit_broadcast_chain",
            category="broadcasting",
            difficulty="advanced",
            title_ja="Chained Broadcasting",
            title_en="Chained Broadcasting",
            cases=[
                ProblemCase(
                    name="Triple Add",
                    description_ja="a [2, 1, 1], b [1, 3, 1], c [1, 1, 4] を足して [2, 3, 4] にしてください。",
                    description_en="Add a [2,1,1], b [1,3,1], c [1,1,4] to get [2,3,4].",
                    hint_ja="a + b + c を使用します。",
                    hint_en="Use a + b + c.",
                    setup_code="""a = torch.randn(2, 1, 1)
b = torch.randn(1, 3, 1)
c = torch.randn(1, 1, 4)""",
                    solution_code="result = a + b + c"
                ),
                ProblemCase(
                    name="Weighted Sum",
                    description_ja="weights [3, 1, 1] と data [1, 4, 5] の加重和を計算してください。",
                    description_en="Compute weighted data using broadcasting.",
                    hint_ja="(weights * data).sum(dim=0) を使用します。",
                    hint_en="Use (weights * data).sum(dim=0).",
                    setup_code="""weights = torch.tensor([0.2, 0.3, 0.5]).view(3, 1, 1)
data = torch.randn(3, 4, 5)""",
                    solution_code="result = (weights * data).sum(dim=0)"
                ),
            ],
            tags=["chain", "broadcasting"],
        ),

        Problem(
            id="broadcast_shapes",
            category="broadcasting",
            difficulty="beginner",
            title_ja="Broadcast Shapes",
            title_en="Broadcast Shapes",
            cases=[
                ProblemCase(
                    name="Infer Shape",
                    description_ja="形状 [3, 1] と [4] のブロードキャスト結果の形状を計算してください。",
                    description_en="Infer broadcast shape of [3, 1] and [4].",
                    hint_ja="torch.broadcast_shapes((3, 1), (4,)) を使用します。",
                    hint_en="Use torch.broadcast_shapes((3, 1), (4,)).",
                    setup_code="",
                    solution_code="result = torch.broadcast_shapes((3, 1), (4,))"
                ),
                ProblemCase(
                    name="Complex Shape",
                    description_ja="形状 [2, 1, 4] と [3, 1] のブロードキャスト結果の形状を計算してください。",
                    description_en="Infer broadcast shape of [2, 1, 4] and [3, 1].",
                    hint_ja="torch.broadcast_shapes を使用します。",
                    hint_en="Use torch.broadcast_shapes.",
                    setup_code="",
                    solution_code="result = torch.broadcast_shapes((2, 1, 4), (3, 1))"
                ),
            ],
            tags=["broadcast_shapes"],
        ),
    ]

    return problems

