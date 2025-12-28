"""Indexing & Slicing problems - advanced tensor access patterns."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_indexing_slicing_problems() -> List[Problem]:
    """Get all Indexing & Slicing category problems."""

    problems = [
        # Beginner level
        Problem(
            id="index_001",
            category="indexing_slicing",
            difficulty="beginner",
            title_ja="基本的なスライシング",
            title_en="Basic Slicing",
            description_ja="形状 [10, 20] のテンソルから最初の5行を取得してください。",
            description_en="Get the first 5 rows from a tensor of shape [10, 20].",
            hint_ja="x[:5] を使用します。",
            hint_en="Use x[:5].",
            setup_code="x = torch.randn(10, 20)",
            solution_code="result = x[:5]",
            tags=["slicing", "basic"],
        ),

        Problem(
            id="index_002",
            category="indexing_slicing",
            difficulty="beginner",
            title_ja="特定の列の抽出",
            title_en="Extract Specific Column",
            description_ja="形状 [10, 20] のテンソルから5番目の列（インデックス4）を取得してください。",
            description_en="Get the 5th column (index 4) from a tensor of shape [10, 20].",
            hint_ja="x[:, 4] を使用します。",
            hint_en="Use x[:, 4].",
            setup_code="x = torch.randn(10, 20)",
            solution_code="result = x[:, 4]",
            tags=["slicing", "column"],
        ),

        Problem(
            id="index_003",
            category="indexing_slicing",
            difficulty="beginner",
            title_ja="負のインデックス",
            title_en="Negative Indexing",
            description_ja="形状 [100] のテンソルから最後の10要素を取得してください。",
            description_en="Get the last 10 elements from a tensor of shape [100].",
            hint_ja="x[-10:] を使用します。",
            hint_en="Use x[-10:].",
            setup_code="x = torch.arange(100)",
            solution_code="result = x[-10:]",
            tags=["slicing", "negative_index"],
        ),

        Problem(
            id="index_004",
            category="indexing_slicing",
            difficulty="beginner",
            title_ja="ステップ付きスライス",
            title_en="Slicing with Step",
            description_ja="形状 [20] のテンソルから2つおきに要素を取得してください。",
            description_en="Get every other element from a tensor of shape [20].",
            hint_ja="x[::2] を使用します。",
            hint_en="Use x[::2].",
            setup_code="x = torch.arange(20)",
            solution_code="result = x[::2]",
            tags=["slicing", "step"],
        ),

        # Intermediate level
        Problem(
            id="index_005",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Boolean マスキング",
            title_en="Boolean Masking",
            description_ja="テンソル x から0より大きい要素のみを取り出してください。",
            description_en="Extract only elements greater than 0 from tensor x.",
            hint_ja="x[x > 0] を使用します。",
            hint_en="Use x[x > 0].",
            setup_code="x = torch.randn(100)",
            solution_code="result = x[x > 0]",
            tags=["boolean", "masking", "conditional"],
        ),

        Problem(
            id="index_006",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="torch.where による条件選択",
            title_en="Conditional Selection with torch.where",
            description_ja="テンソル x の要素が0より大きい場合はそのまま、それ以外は0にしてください。",
            description_en="Keep elements of tensor x if they are greater than 0, otherwise set to 0.",
            hint_ja="torch.where(x > 0, x, torch.zeros_like(x)) を使用します。",
            hint_en="Use torch.where(x > 0, x, torch.zeros_like(x)).",
            setup_code="x = torch.randn(10, 10)",
            solution_code="result = torch.where(x > 0, x, torch.zeros_like(x))",
            tags=["where", "conditional"],
        ),

        Problem(
            id="index_007",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Fancy Indexing - 複数インデックス",
            title_en="Fancy Indexing - Multiple Indices",
            description_ja="形状 [100] のテンソルからインデックス [5, 10, 15, 20, 25] の要素を取得してください。",
            description_en="Get elements at indices [5, 10, 15, 20, 25] from a tensor of shape [100].",
            hint_ja="x[[5, 10, 15, 20, 25]] または x[torch.tensor([5, 10, 15, 20, 25])] を使用します。",
            hint_en="Use x[[5, 10, 15, 20, 25]] or x[torch.tensor([5, 10, 15, 20, 25])].",
            setup_code="x = torch.arange(100)",
            solution_code="result = x[torch.tensor([5, 10, 15, 20, 25])]",
            tags=["fancy_indexing", "multi_index"],
        ),

        Problem(
            id="index_008",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="2D Fancy Indexing",
            title_en="2D Fancy Indexing",
            description_ja="形状 [10, 20] のテンソルから、行インデックス [0, 2, 4] と列インデックス [1, 3, 5] の要素を対応させて取得してください。",
            description_en="Get elements at row indices [0, 2, 4] and column indices [1, 3, 5] (paired) from a tensor of shape [10, 20].",
            hint_ja="x[[0, 2, 4], [1, 3, 5]] または x[torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5])] を使用します。",
            hint_en="Use x[[0, 2, 4], [1, 3, 5]] or x[torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5])].",
            setup_code="x = torch.randn(10, 20)",
            solution_code="result = x[torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5])]",
            tags=["fancy_indexing", "2d"],
        ),

        Problem(
            id="index_009",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Ellipsis の使用",
            title_en="Using Ellipsis",
            description_ja="形状 [32, 3, 224, 224] のテンソルから、全バッチの赤チャネル（インデックス0）のみを取得してください。結果の形状は [32, 224, 224] です。",
            description_en="Get only the red channel (index 0) for all batches from a tensor of shape [32, 3, 224, 224]. Result shape is [32, 224, 224].",
            hint_ja="x[:, 0, :, :] または x[:, 0] を使用します。",
            hint_en="Use x[:, 0, :, :] or x[:, 0].",
            setup_code="x = torch.randn(32, 3, 224, 224)",
            solution_code="result = x[:, 0]",
            tags=["slicing", "ellipsis", "cv"],
        ),

        Problem(
            id="index_010",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="マルチディメンションスライス",
            title_en="Multi-dimensional Slice",
            description_ja="形状 [32, 128, 512] のテンソルから、全バッチの最初の64トークンの最初の256次元を取得してください。",
            description_en="Get the first 64 tokens and first 256 dimensions for all batches from a tensor of shape [32, 128, 512].",
            hint_ja="x[:, :64, :256] を使用します。",
            hint_en="Use x[:, :64, :256].",
            setup_code="x = torch.randn(32, 128, 512)",
            solution_code="result = x[:, :64, :256]",
            tags=["slicing", "multi_dim", "nlp"],
        ),

        # Advanced level
        Problem(
            id="index_011",
            category="indexing_slicing",
            difficulty="advanced",
            title_ja="複雑な Boolean マスク",
            title_en="Complex Boolean Mask",
            description_ja="形状 [10, 20] のテンソルから、各行の平均より大きい要素のみを含む1Dテンソルを作成してください。",
            description_en="Create a 1D tensor containing only elements greater than their row's mean from a tensor of shape [10, 20].",
            hint_ja="行ごとの平均を計算し、ブロードキャストして比較します。",
            hint_en="Calculate row-wise mean and broadcast for comparison.",
            setup_code="x = torch.randn(10, 20)",
            solution_code="""mean = x.mean(dim=1, keepdim=True)
mask = x > mean
result = x[mask]""",
            tags=["boolean", "masking", "mean", "broadcasting"],
        ),

        Problem(
            id="index_012",
            category="indexing_slicing",
            difficulty="advanced",
            title_ja="nonzero を使った抽出",
            title_en="Extraction with nonzero",
            description_ja="形状 [100] のテンソルから0でない要素のインデックスを取得し、それらの要素を抽出してください。",
            description_en="Get indices of non-zero elements from a tensor of shape [100] and extract those elements.",
            hint_ja="torch.nonzero() を使ってインデックスを取得します。",
            hint_en="Use torch.nonzero() to get indices.",
            setup_code="""x = torch.randn(100)
x[x < 0.5] = 0  # Some elements become 0""",
            solution_code="""indices = torch.nonzero(x, as_tuple=True)[0]
result = x[indices]""",
            tags=["nonzero", "sparse"],
        ),

        Problem(
            id="index_013",
            category="indexing_slicing",
            difficulty="advanced",
            title_ja="バッチごとに異なるインデックスで抽出",
            title_en="Extract Different Indices per Batch",
            description_ja="形状 [32, 100] のテンソルから、各バッチのインデックス indices (形状 [32]) で指定された要素を取得してください。結果の形状は [32] です。",
            description_en="Get elements at indices specified by 'indices' (shape [32]) for each batch from a tensor of shape [32, 100]. Result shape is [32].",
            hint_ja="x[torch.arange(32), indices] を使用します。",
            hint_en="Use x[torch.arange(32), indices].",
            setup_code="""x = torch.randn(32, 100)
indices = torch.randint(0, 100, (32,))""",
            solution_code="result = x[torch.arange(32), indices]",
            tags=["fancy_indexing", "batch", "arange"],
        ),

        Problem(
            id="index_014",
            category="indexing_slicing",
            difficulty="advanced",
            title_ja="Top-K インデックスによる抽出",
            title_en="Extract Top-K Indices",
            description_ja="形状 [32, 1000] のテンソルから、各バッチの上位5つの値を取得してください。結果の形状は [32, 5] です。",
            description_en="Get the top 5 values for each batch from a tensor of shape [32, 1000]. Result shape is [32, 5].",
            hint_ja="torch.topk() を使用します。",
            hint_en="Use torch.topk().",
            setup_code="x = torch.randn(32, 1000)",
            solution_code="""result, _ = torch.topk(x, k=5, dim=1)""",
            tags=["topk", "sorting"],
        ),

        # Expert level
        Problem(
            id="index_015",
            category="indexing_slicing",
            difficulty="expert",
            title_ja="Attention マスクの適用",
            title_en="Apply Attention Mask",
            description_ja="形状 [32, 8, 128, 128] のアテンションスコアに、形状 [32, 1, 1, 128] のマスク（0または1）を適用してください。マスクが0の位置は-inf、1の位置はそのままにします。",
            description_en="Apply a mask of shape [32, 1, 1, 128] (0 or 1) to attention scores of shape [32, 8, 128, 128]. Set -inf where mask is 0, keep original where mask is 1.",
            hint_ja="torch.where() を使い、マスクが0の位置に-infを設定します。",
            hint_en="Use torch.where() to set -inf where mask is 0.",
            setup_code="""scores = torch.randn(32, 8, 128, 128)
mask = torch.randint(0, 2, (32, 1, 1, 128))""",
            solution_code="""result = torch.where(mask == 1, scores, torch.tensor(float('-inf')))""",
            tags=["where", "masking", "attention", "nlp"],
        ),
    ]

    return problems
