"""Gather & Scatter problems - advanced indexing and accumulation."""

from typing import List
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_gather_scatter_problems() -> List[Problem]:
    """Get all Gather & Scatter category problems."""

    problems = [
        Problem(
            id="gather_basics",
            category="gather_scatter",
            difficulty="intermediate",
            title_ja="Gather Basics",
            title_en="Gather Basics",
            cases=[
                ProblemCase(
                    name="Gather (dim=0)",
                    description_ja="テンソル x [4, 5] から dim=0 に沿って indices [2, 5] の要素を集めてください。",
                    description_en="Gather elements from x along dim 0 using indices.",
                    hint_ja="torch.gather(x, 0, indices) を使用します。",
                    hint_en="Use torch.gather(x, 0, indices).",
                    setup_code="""x = torch.randn(4, 5)
indices = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])""",
                    solution_code="result = torch.gather(x, 0, indices)"
                ),
                ProblemCase(
                    name="Gather (dim=1)",
                    description_ja="テンソル x [3, 4] から dim=1 に沿って要素を集めてください。",
                    description_en="Gather elements from x along dim 1.",
                    hint_ja="torch.gather(x, 1, indices) を使用します。",
                    hint_en="Use torch.gather(x, 1, indices).",
                    setup_code="""x = torch.randn(3, 4)
indices = torch.tensor([[0, 1], [1, 2], [2, 3]])""",
                    solution_code="result = torch.gather(x, 1, indices)"
                ),
                ProblemCase(
                    name="take_along_dim",
                    description_ja="argsortの結果を使って、x を値の小さい順に並べ替えてください (dim=0)。",
                    description_en="Sort x dim=0 using take_along_dim.",
                    hint_ja="torch.take_along_dim(x, sorted_indices, dim=0) を使用します。",
                    hint_en="Use torch.take_along_dim(x, sorted_indices, dim=0).",
                    setup_code="""x = torch.randn(4, 5)
sorted_indices = torch.argsort(x, dim=0)""",
                    solution_code="result = torch.take_along_dim(x, sorted_indices, dim=0)"
                ),
            ],
            tags=["gather", "take_along_dim"],
        ),

        Problem(
            id="scatter_accumulation",
            category="gather_scatter",
            difficulty="advanced",
            title_ja="Scatter Accumulation & Reduction",
            title_en="Scatter Accumulation & Reduction",
            cases=[
                ProblemCase(
                    name="Scatter Add (1D)",
                    description_ja="ゼロ初期化された x [10] に、indices の位置へ src の値を加算してください。",
                    description_en="Scatter add src into x at indices.",
                    hint_ja="x.scatter_add(0, indices, src) を使用します。",
                    hint_en="Use x.scatter_add(0, indices, src).",
                    setup_code="""x = torch.zeros(10)
src = torch.ones(5)
indices = torch.tensor([0, 1, 2, 0, 1])""",
                    solution_code="result = x.scatter_add(0, indices, src)"
                ),
                ProblemCase(
                    name="Scatter Add (2D)",
                    description_ja="ゼロ初期化された x [5, 5] に、dim=1 方向に src を加算してください。",
                    description_en="Scatter add src into x along dim 1.",
                    hint_ja="x.scatter_add(1, indices, src) を使用します。",
                    hint_en="Use x.scatter_add(1, indices, src).",
                    setup_code="""x = torch.zeros(5, 5)
src = torch.randn(5, 3)
indices = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]])""",
                    solution_code="result = x.scatter_add(1, indices, src)"
                ),
                ProblemCase(
                    name="Scatter Reduce (sum)",
                    description_ja="scatter_reduce を使って合計を計算してください (reduce='sum')。",
                    description_en="Compute sum using scatter_reduce.",
                    hint_ja="x.scatter_reduce(0, indices, src, reduce='sum') を使用します。",
                    hint_en="Use x.scatter_reduce(0, indices, src, reduce='sum').",
                    setup_code="""x = torch.zeros(10)
src = torch.ones(6)
indices = torch.tensor([0, 1, 2, 0, 1, 2])""",
                    solution_code="result = x.scatter_reduce(0, indices, src, reduce='sum', include_self=True)"
                ),
                ProblemCase(
                    name="Scatter Reduce (mean)",
                    description_ja="scatter_reduce を使って平均を計算してください (reduce='mean')。",
                    description_en="Compute mean using scatter_reduce.",
                    hint_ja="x.scatter_reduce(0, indices, src, reduce='mean', include_self=False) を使用します。",
                    hint_en="Use x.scatter_reduce(0, indices, src, reduce='mean', include_self=False).",
                    setup_code="""x = torch.zeros(10)
src = torch.ones(6) * 10
indices = torch.tensor([0, 1, 2, 0, 1, 2])""",
                    solution_code="result = x.scatter_reduce(0, indices, src, reduce='mean', include_self=False)"
                ),
            ],
            tags=["scatter_add", "scatter_reduce"],
        ),

        Problem(
            id="gather_scatter_applications",
            category="gather_scatter",
            difficulty="expert",
            title_ja="Gather & Scatter Applications",
            title_en="Gather & Scatter Applications",
            cases=[
                ProblemCase(
                    name="One-hot Encoding",
                    description_ja="クラスインデックス labels [N] を One-hot ベクトル [N, C] に変換してください (C=5)。",
                    description_en="Convert labels [N] to one-hot [N, C] (C=5).",
                    hint_ja="torch.zeros(N, 5).scatter_(1, labels.unsqueeze(1), 1.0) を使用します。",
                    hint_en="Use torch.zeros(N, 5).scatter_(1, labels.unsqueeze(1), 1.0).",
                    setup_code="""N = 4
C = 5
labels = torch.tensor([0, 2, 4, 1])""",
                    solution_code="result = torch.zeros(N, C).scatter_(1, labels.unsqueeze(1), 1.0)"
                ),
                ProblemCase(
                    name="Batch Gathering",
                    description_ja="バッチデータ batch [B, N, D] から、各バッチごとに指定されたインデックス indices [B, K] の要素を取得し [B, K, D] にしてください。",
                    description_en="Gather elements from batch [B, N, D] using indices [B, K] to get [B, K, D].",
                    hint_ja="indices.unsqueeze(-1).expand(-1, -1, D) でインデックスを拡張し、gatherします。",
                    hint_en="Expand indices and gather.",
                    setup_code="""B, N, D = 2, 5, 3
K = 2
batch = torch.randn(B, N, D)
indices = torch.tensor([[0, 1], [3, 4]])""",
                    solution_code="""expanded_indices = indices.unsqueeze(-1).expand(-1, -1, D)
result = torch.gather(batch, 1, expanded_indices)"""
                ),
            ],
            tags=["one_hot", "scatter", "gather", "batch"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="embedding_style_gather",
            category="gather_scatter",
            difficulty="intermediate",
            title_ja="Embedding-Style Gather",
            title_en="Embedding-Style Gather",
            cases=[
                ProblemCase(
                    name="Simple Embedding",
                    description_ja="埋め込み行列 emb [V, D] からインデックス idx [N] の行を取得してください。",
                    description_en="Get rows from embedding emb using idx.",
                    hint_ja="emb[idx] または torch.embedding(emb, idx) を使用します。",
                    hint_en="Use emb[idx] or F.embedding(emb, idx).",
                    setup_code="""V, D = 100, 32
N = 5
emb = torch.randn(V, D)
idx = torch.tensor([10, 20, 30, 40, 50])""",
                    solution_code="result = emb[idx]"
                ),
                ProblemCase(
                    name="Batch Embedding",
                    description_ja="埋め込み行列 emb [V, D] からバッチインデックス idx [B, L] の行を取得して [B, L, D] にしてください。",
                    description_en="Get batch embeddings of shape [B, L, D].",
                    hint_ja="emb[idx] を使用します。",
                    hint_en="Use emb[idx].",
                    setup_code="""V, D = 100, 32
B, L = 4, 10
emb = torch.randn(V, D)
idx = torch.randint(0, V, (B, L))""",
                    solution_code="result = emb[idx]"
                ),
            ],
            tags=["embedding", "gather"],
        ),

        Problem(
            id="scatter_max_min",
            category="gather_scatter",
            difficulty="advanced",
            title_ja="Scatter Max and Min",
            title_en="Scatter Max and Min",
            cases=[
                ProblemCase(
                    name="Scatter Reduce Max",
                    description_ja="scatter_reduce で各位置の最大値を計算してください。",
                    description_en="Compute max at each position using scatter_reduce.",
                    hint_ja="x.scatter_reduce(0, indices, src, reduce='amax') を使用します。",
                    hint_en="Use x.scatter_reduce with reduce='amax'.",
                    setup_code="""x = torch.full((5,), float('-inf'))
src = torch.tensor([1., 3., 2., 5., 4., 6.])
indices = torch.tensor([0, 0, 1, 1, 2, 2])""",
                    solution_code="result = x.scatter_reduce(0, indices, src, reduce='amax', include_self=False)"
                ),
                ProblemCase(
                    name="Scatter Reduce Min",
                    description_ja="scatter_reduce で各位置の最小値を計算してください。",
                    description_en="Compute min at each position using scatter_reduce.",
                    hint_ja="x.scatter_reduce(0, indices, src, reduce='amin') を使用します。",
                    hint_en="Use x.scatter_reduce with reduce='amin'.",
                    setup_code="""x = torch.full((5,), float('inf'))
src = torch.tensor([1., 3., 2., 5., 4., 6.])
indices = torch.tensor([0, 0, 1, 1, 2, 2])""",
                    solution_code="result = x.scatter_reduce(0, indices, src, reduce='amin', include_self=False)"
                ),
            ],
            tags=["scatter_reduce", "max", "min"],
        ),

        Problem(
            id="index_select_multi",
            category="gather_scatter",
            difficulty="intermediate",
            title_ja="Multi-Dim Index Select",
            title_en="Multi-Dim Index Select",
            cases=[
                ProblemCase(
                    name="Select Rows",
                    description_ja="テンソル x [10, 5] から行 [0, 2, 5, 9] を選択してください。",
                    description_en="Select rows [0, 2, 5, 9] from x.",
                    hint_ja="torch.index_select(x, 0, indices) を使用します。",
                    hint_en="Use torch.index_select(x, 0, indices).",
                    setup_code="""x = torch.randn(10, 5)
indices = torch.tensor([0, 2, 5, 9])""",
                    solution_code="result = torch.index_select(x, 0, indices)"
                ),
                ProblemCase(
                    name="Select Cols",
                    description_ja="テンソル x [5, 10] から列 [1, 3, 7] を選択してください。",
                    description_en="Select columns [1, 3, 7] from x.",
                    hint_ja="torch.index_select(x, 1, indices) を使用します。",
                    hint_en="Use torch.index_select(x, 1, indices).",
                    setup_code="""x = torch.randn(5, 10)
indices = torch.tensor([1, 3, 7])""",
                    solution_code="result = torch.index_select(x, 1, indices)"
                ),
            ],
            tags=["index_select"],
        ),

        Problem(
            id="knn_style_gather",
            category="gather_scatter",
            difficulty="expert",
            title_ja="KNN-Style Gather",
            title_en="KNN-Style Gather",
            cases=[
                ProblemCase(
                    name="K Nearest Points",
                    description_ja="点群 points [N, D] から、各点について K 個の最近傍を集めてください (knn_idx 使用)。",
                    description_en="Gather K nearest neighbors for each point.",
                    hint_ja="gather で knn_idx を使います。",
                    hint_en="Use gather with knn_idx.",
                    setup_code="""N, D, K = 100, 3, 5
points = torch.randn(N, D)
knn_idx = torch.randint(0, N, (N, K))""",
                    solution_code="result = points[knn_idx]"
                ),
                ProblemCase(
                    name="Batch KNN",
                    description_ja="バッチ点群 points [B, N, D] と knn_idx [B, N, K] から K 近傍を集めて [B, N, K, D] にしてください。",
                    description_en="Gather K neighbors with batch.",
                    hint_ja="gather with expanded indices.",
                    hint_en="Use gather with expanded indices.",
                    setup_code="""B, N, D, K = 2, 50, 3, 5
points = torch.randn(B, N, D)
knn_idx = torch.randint(0, N, (B, N, K))""",
                    solution_code="""expanded_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
result = points.unsqueeze(2).expand(-1, -1, K, -1).gather(1, expanded_idx)"""
                ),
            ],
            tags=["knn", "gather"],
        ),

        Problem(
            id="scatter_softmax",
            category="gather_scatter",
            difficulty="expert",
            title_ja="Scatter-Based Softmax",
            title_en="Scatter-Based Softmax",
            cases=[
                ProblemCase(
                    name="Grouped Softmax",
                    description_ja="グループごとにソフトマックスを計算してください (scatter を使用)。",
                    description_en="Compute softmax within groups using scatter.",
                    hint_ja="scatter_reduce で max と sum を使います。",
                    hint_en="Use scatter_reduce for max and sum.",
                    setup_code="""src = torch.tensor([1., 2., 3., 1., 2.])
indices = torch.tensor([0, 0, 0, 1, 1])
num_groups = 2""",
                    solution_code="""max_vals = torch.zeros(num_groups).scatter_reduce(0, indices, src, reduce='amax', include_self=False)
centered = src - max_vals[indices]
exp_src = centered.exp()
sum_exp = torch.zeros(num_groups).scatter_add(0, indices, exp_src)
result = exp_src / sum_exp[indices]"""
                ),
            ],
            tags=["scatter", "softmax"],
        ),

        Problem(
            id="gather_along_axis",
            category="gather_scatter",
            difficulty="intermediate",
            title_ja="Gather Along Axis",
            title_en="Gather Along Axis",
            cases=[
                ProblemCase(
                    name="Max Value Gather",
                    description_ja="テンソル x [4, 5] の各行の最大値を gather で取得してください。",
                    description_en="Gather max value per row.",
                    hint_ja="argmax と gather を組み合わせます。",
                    hint_en="Combine argmax and gather.",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = torch.gather(x, 1, x.argmax(dim=1, keepdim=True))"
                ),
                ProblemCase(
                    name="Sorted Elements",
                    description_ja="テンソル x [3, 6] の各行をソートした値を gather で取得してください。",
                    description_en="Get sorted elements per row using gather.",
                    hint_ja="argsort と gather を組み合わせます。",
                    hint_en="Combine argsort and gather.",
                    setup_code="x = torch.randn(3, 6)",
                    solution_code="result = torch.gather(x, 1, x.argsort(dim=1))"
                ),
            ],
            tags=["gather", "axis"],
        ),

        Problem(
            id="sparse_gather_patterns",
            category="gather_scatter",
            difficulty="advanced",
            title_ja="Sparse Gather Patterns",
            title_en="Sparse Gather Patterns",
            cases=[
                ProblemCase(
                    name="Sparse to Dense",
                    description_ja="スパースインデックス sparse_idx [K] と値 values [K] から密なテンソル [N] を作成してください。",
                    description_en="Create dense tensor from sparse indices and values.",
                    hint_ja="scatter を使用します。",
                    hint_en="Use scatter.",
                    setup_code="""N = 10
K = 4
sparse_idx = torch.tensor([1, 3, 5, 7])
values = torch.tensor([1., 2., 3., 4.])""",
                    solution_code="result = torch.zeros(N).scatter(0, sparse_idx, values)"
                ),
                ProblemCase(
                    name="Dense to Sparse",
                    description_ja="密なテンソル dense [N] から非ゼロインデックスの値を抽出してください。",
                    description_en="Extract non-zero values from dense tensor.",
                    hint_ja="nonzero と gather を使用します。",
                    hint_en="Use nonzero and indexing.",
                    setup_code="""dense = torch.tensor([0., 1., 0., 2., 0., 3.])""",
                    solution_code="result = dense[dense.nonzero(as_tuple=True)]"
                ),
            ],
            tags=["sparse", "gather", "scatter"],
        ),
    ]

    return problems

