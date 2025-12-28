"""Gather & Scatter problems - index-based collection and distribution."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_gather_scatter_problems() -> List[Problem]:
    """Get all Gather & Scatter category problems."""

    problems = [
        # Beginner level
        Problem(
            id="gather_001",
            category="gather_scatter",
            difficulty="beginner",
            title_ja="基本的な gather",
            title_en="Basic gather",
            description_ja="形状 [5, 10] のテンソルと、形状 [5, 3] のインデックステンソルを使って、各行から3つの要素を収集してください。",
            description_en="Use torch.gather to collect 3 elements from each row using a tensor of shape [5, 10] and index tensor of shape [5, 3].",
            hint_ja="torch.gather(input, dim=1, index) を使用します。",
            hint_en="Use torch.gather(input, dim=1, index).",
            setup_code="""x = torch.randn(5, 10)
indices = torch.tensor([[0, 2, 5], [1, 3, 7], [0, 4, 9], [2, 5, 8], [1, 6, 9]])""",
            solution_code="result = torch.gather(x, dim=1, index=indices)",
            tags=["gather", "basic"],
        ),

        Problem(
            id="gather_002",
            category="gather_scatter",
            difficulty="beginner",
            title_ja="1D テンソルの scatter",
            title_en="1D Tensor scatter",
            description_ja="形状 [10] のゼロテンソルのインデックス [2, 5機, 8] に値 [1.0, 2.0, 3.0] を設定してください。",
            description_en="Set values [1.0, 2.0, 3.0] at indices [2, 5, 8] in a zero tensor of shape [10].",
            hint_ja="scatter_(dim, index, src) を使用します。",
            hint_en="Use scatter_(dim, index, src).",
            setup_code="""result = torch.zeros(10)
indices = torch.tensor([2, 5, 8])
values = torch.tensor([1.0, 2.0, 3.0])""",
            solution_code="result = result.scatter_(0, indices, values)",
            tags=["scatter", "1d"],
        ),

        Problem(
            id="gather_003",
            category="gather_scatter",
            difficulty="intermediate",
            title_ja="クラス確率の収集",
            title_en="Gather Class Probabilities",
            description_ja="形状 [32, 1000] のロジットテンソルから、形状 [32] の正解ラベルに対応する確率を収集してください。結果の形状は [32, 1] です。",
            description_en="Gather probabilities corresponding to ground truth labels (shape [32]) from logits of shape [32, 1000]. Result shape is [32, 1].",
            hint_ja="labels を unsqueeze して gather します。",
            hint_en="Use unsqueeze on labels and gather.",
            setup_code="""logits = torch.randn(32, 1000)
labels = torch.randint(0, 1000, (32,))""",
            solution_code="result = torch.gather(logits, dim=1, index=labels.unsqueeze(1))",
            tags=["gather", "classification", "ml"],
        ),

        Problem(
            id="gather_004",
            category="gather_scatter",
            difficulty="intermediate",
            title_ja="One-hot エンコーディング (scatter)",
            title_en="One-hot Encoding (scatter)",
            description_ja="形状 [32] のラベルテンソル（値は0~9）を、形状 [32, 10] の one-hot テンソルに変換してください。",
            description_en="Convert a label tensor of shape [32] (values 0-9) to a one-hot tensor of shape [32, 10].",
            hint_ja="zeros テンソルを作り、scatter_ で 1 を設定します。",
            hint_en="Create a zeros tensor and use scatter_ to set 1s.",
            setup_code="labels = torch.randint(0, 10, (32,))",
            solution_code="""result = torch.zeros(32, 10)
result = result.scatter_(1, labels.unsqueeze(1), 1)""",
            tags=["scatter", "one_hot", "ml"],
        ),

        Problem(
            id="gather_005",
            category="gather_scatter",
            difficulty="intermediate",
            title_ja="Embedding の Gather",
            title_en="Embedding Gather",
            description_ja="形状 [10000, 512] の埋め込み行列から、形状 [32, 128] のトークンIDに対応する埋め込みを取得してください。結果の形状は [32, 128, 512] です。",
            description_en="Get embeddings corresponding to token IDs of shape [32, 128] from an embedding matrix of shape [10000, 512]. Result shape is [32, 128, 512].",
            hint_ja="embedding[token_ids] で取得できます。",
            hint_en="Use embedding[token_ids] to retrieve.",
            setup_code="""embedding = torch.randn(10000, 512)
token_ids = torch.randint(0, 10000, (32, 128))""",
            solution_code="result = embedding[token_ids]",
            tags=["gather", "embedding", "nlp"],
        ),

        Problem(
            id="gather_006",
            category="gather_scatter",
            difficulty="advanced",
            title_ja="Scatter Add による集約",
            title_en="Aggregation with Scatter Add",
            description_ja="形状 [100] の値テンソルを、形状 [100] のインデックス（0~9の範囲）でグループ化して合計し、形状 [10] の結果を作成してください。",
            description_en="Group and sum a value tensor of shape [100] by indices (range 0-9) of shape [100] to create a result of shape [10].",
            hint_ja="scatter_add_ を使用します。",
            hint_en="Use scatter_add_.",
            setup_code="""values = torch.randn(100)
indices = torch.randint(0, 10, (100,))""",
            solution_code="""result = torch.zeros(10)
result = result.scatter_add_(0, indices, values)""",
            tags=["scatter_add", "aggregation"],
        ),

        Problem(
            id="gather_007",
            category="gather_scatter",
            difficulty="advanced",
            title_ja="バッチごとの Top-K Gather",
            title_en="Batch-wise Top-K Gather",
            description_ja="形状 [32, 1000] のテンソルから、各バッチの上位5つのインデックスを取得し、その値を収集してください。結果の形状は [32, 5] です。",
            description_en="Get top-5 indices for each batch from a tensor of shape [32, 1000] and gather those values. Result shape is [32, 5].",
            hint_ja="topk でインデックスを取得し、gather します。",
            hint_en="Use topk to get indices, then gather.",
            setup_code="x = torch.randn(32, 1000)",
            solution_code="""_, indices = torch.topk(x, k=5, dim=1)
result = torch.gather(x, dim=1, index=indices)""",
            tags=["gather", "topk", "batch"],
        ),

        Problem(
            id="gather_008",
            category="gather_scatter",
            difficulty="advanced",
            title_ja="3D テンソルの Scatter",
            title_en="3D Tensor Scatter",
            description_ja="形状 [10, 20, 30] のゼロテンソルに、形状 [10, 5, 30] のインデックスと値を使って scatter してください。",
            description_en="Scatter values to a zero tensor of shape [10, 20, 30] using indices and values of shape [10, 5, 30].",
            hint_ja="scatter_(dim=1, index, src) を使用します。",
            hint_en="Use scatter_(dim=1, index, src).",
            setup_code="""result = torch.zeros(10, 20, 30)
indices = torch.randint(0, 20, (10, 5, 30))
values = torch.randn(10, 5, 30)""",
            solution_code="result = result.scatter_(1, indices, values)",
            tags=["scatter", "3d"],
        ),

        Problem(
            id="gather_009",
            category="gather_scatter",
            difficulty="expert",
            title_ja="Attention での Gather（KV キャッシュ）",
            title_en="Gather in Attention (KV Cache)",
            description_ja="形状 [32, 1000, 64] のキャッシュから、形状 [32, 128] の位置インデックスを使って値を収集してください。結果の形状は [32, 128, 64] です。",
            description_en="Gather values from a cache of shape [32, 1000, 64] using position indices of shape [32, 128]. Result shape is [32, 128, 64].",
            hint_ja="indices を [32, 128, 64] に expand してから gather します。",
            hint_en="Expand indices to [32, 128, 64] before gathering.",
            setup_code="""cache = torch.randn(32, 1000, 64)
positions = torch.randint(0, 1000, (32, 128))""",
            solution_code="""indices = positions.unsqueeze(-1).expand(-1, -1, 64)
result = torch.gather(cache, dim=1, index=indices)""",
            tags=["gather", "attention", "cache", "nlp"],
        ),

        Problem(
            id="gather_010",
            category="gather_scatter",
            difficulty="expert",
            title_ja="Scatter によるヒストグラム作成",
            title_en="Histogram Creation with Scatter",
            description_ja="形状 [1000] の値（0.0~1.0の範囲）を10個のビンに分類し、各ビンのカウントを含む形状 [10] のヒストグラムを作成してください。",
            description_en="Classify values of shape [1000] (range 0.0-1.0) into 10 bins and create a histogram of shape [10] with counts for each bin.",
            hint_ja="値をビンのインデックスに変換し、scatter_add_ を使います。",
            hint_en="Convert values to bin indices and use scatter_add_.",
            setup_code="values = torch.rand(1000)",
            solution_code="""bin_indices = (values * 10).long().clamp(0, 9)
ones = torch.ones(1000)
result = torch.zeros(10)
result = result.scatter_add_(0, bin_indices, ones)""",
            tags=["scatter_add", "histogram", "statistics"],
        ),
    ]

    return problems
