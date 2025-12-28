"""Stacking & Splitting problems - combining and separating tensors."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_stacking_splitting_problems() -> List[Problem]:
    """Get all Stacking & Splitting category problems."""

    problems = [
        # Beginner level
        Problem(
            id="stack_001",
            category="stacking_splitting",
            difficulty="beginner",
            title_ja="リストのテンソルを stack",
            title_en="Stack List of Tensors",
            description_ja="3つの形状 [10, 20] のテンソルを新しい次元（dim=0）で積み重ねて [3, 10, 20] にしてください。",
            description_en="Stack three tensors of shape [10, 20] along a new dimension (dim=0) to get [3, 10, 20].",
            hint_ja="torch.stack([a, b, c], dim=0) を使用します。",
            hint_en="Use torch.stack([a, b, c], dim=0).",
            setup_code="""a = torch.randn(10, 20)
b = torch.randn(10, 20)
c = torch.randn(10, 20)""",
            solution_code="result = torch.stack([a, b, c], dim=0)",
            tags=["stack", "dim0"],
        ),

        Problem(
            id="stack_002",
            category="stacking_splitting",
            difficulty="beginner",
            title_ja="リストのテンソルを cat",
            title_en="Concatenate List of Tensors",
            description_ja="3つの形状 [10, 20] のテンソルを dim=0 で結合して [30, 20] にしてください。",
            description_en="Concatenate three tensors of shape [10, 20] along dim=0 to get [30, 20].",
            hint_ja="torch.cat([a, b, c], dim=0) を使用します。",
            hint_en="Use torch.cat([a, b, c], dim=0).",
            setup_code="""a = torch.randn(10, 20)
b = torch.randn(10, 20)
c = torch.randn(10, 20)""",
            solution_code="result = torch.cat([a, b, c], dim=0)",
            tags=["cat", "concatenate"],
        ),

        Problem(
            id="stack_003",
            category="stacking_splitting",
            difficulty="beginner",
            title_ja="テンソルを等分割",
            title_en="Split Tensor into Equal Parts",
            description_ja="形状 [60, 20] のテンソルを dim=0 で3つに分割してください。それぞれのサイズは [20, 20] です。",
            description_en="Split a tensor of shape [60, 20] into 3 parts along dim=0. Each part has shape [20, 20].",
            hint_ja="torch.chunk(x, chunks=3, dim=0) を使い、リストから要素を取り出します。",
            hint_en="Use torch.chunk(x, chunks=3, dim=0) and extract elements from the list.",
            setup_code="x = torch.randn(60, 20)",
            solution_code="""chunks = torch.chunk(x, chunks=3, dim=0)
result = torch.stack(chunks, dim=0)""",
            tags=["chunk", "split"],
        ),

        Problem(
            id="stack_004",
            category="stacking_splitting",
            difficulty="beginner",
            title_ja="異なるサイズで分割",
            title_en="Split with Different Sizes",
            description_ja="形状 [100] のテンソルを [30, 40, 30] のサイズに分割してください。",
            description_en="Split a tensor of shape [100] into sizes [30, 40, 30].",
            hint_ja="torch.split(x, [30, 40, 30], dim=0) を使用します。",
            hint_en="Use torch.split(x, [30, 40, 30], dim=0).",
            setup_code="x = torch.arange(100)",
            solution_code="""parts = torch.split(x, [30, 40, 30], dim=0)
result = torch.stack([parts[0], parts[1], parts[2]], dim=0)""",
            tags=["split", "variable_size"],
        ),

        # Intermediate level
        Problem(
            id="stack_005",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="次元1での stack",
            title_en="Stack along dim=1",
            description_ja="3つの形状 [10, 20] のテンソルを dim=1 で積み重ねて [10, 3, 20] にしてください。",
            description_en="Stack three tensors of shape [10, 20] along dim=1 to get [10, 3, 20].",
            hint_ja="torch.stack([a, b, c], dim=1) を使用します。",
            hint_en="Use torch.stack([a, b, c], dim=1).",
            setup_code="""a = torch.randn(10, 20)
b = torch.randn(10, 20)
c = torch.randn(10, 20)""",
            solution_code="result = torch.stack([a, b, c], dim=1)",
            tags=["stack", "dim1"],
        ),

        Problem(
            id="stack_006",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="マルチヘッドの結合",
            title_en="Concatenate Multi-Head",
            description_ja="8個の形状 [32, 128, 64] のヘッドテンソルを dim=-1 で結合して [32, 128, 512] にしてください。",
            description_en="Concatenate 8 head tensors of shape [32, 128, 64] along dim=-1 to get [32, 128, 512].",
            hint_ja="torch.cat(heads, dim=-1) を使用します。",
            hint_en="Use torch.cat(heads, dim=-1).",
            setup_code="heads = [torch.randn(32, 128, 64) for _ in range(8)]",
            solution_code="result = torch.cat(heads, dim=-1)",
            tags=["cat", "multi_head", "attention"],
        ),

        Problem(
            id="stack_007",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="マルチヘッドの分割",
            title_en="Split Multi-Head",
            description_ja="形状 [32, 128, 512] のテンソルを8個のヘッドに分割してください。各ヘッドの形状は [32, 128, 64] です。",
            description_en="Split a tensor of shape [32, 128, 512] into 8 heads. Each head has shape [32, 128, 64].",
            hint_ja="torch.chunk(x, chunks=8, dim=-1) を使用します。",
            hint_en="Use torch.chunk(x, chunks=8, dim=-1).",
            setup_code="x = torch.randn(32, 128, 512)",
            solution_code="""heads = torch.chunk(x, chunks=8, dim=-1)
result = torch.stack(heads, dim=0)""",
            tags=["chunk", "multi_head", "attention"],
        ),

        Problem(
            id="stack_008",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="Skip connection の結合",
            title_en="Concatenate Skip Connection",
            description_ja="形状 [32, 64, 56, 56] の特徴マップと形状 [32, 128, 56, 56] のスキップ接続を dim=1 で結合してください。結果の形状は [32, 192, 56, 56] です。",
            description_en="Concatenate a feature map of shape [32, 64, 56, 56] with a skip connection of shape [32, 128, 56, 56] along dim=1. Result shape is [32, 192, 56, 56].",
            hint_ja="torch.cat([feature, skip], dim=1) を使用します。",
            hint_en="Use torch.cat([feature, skip], dim=1).",
            setup_code="""feature = torch.randn(32, 64, 56, 56)
skip = torch.randn(32, 128, 56, 56)""",
            solution_code="result = torch.cat([feature, skip], dim=1)",
            tags=["cat", "skip_connection", "cv"],
        ),

        # Advanced level
        Problem(
            id="stack_009",
            category="stacking_splitting",
            difficulty="advanced",
            title_ja="不均等な分割と再結合",
            title_en="Uneven Split and Recombine",
            description_ja="形状 [32, 100, 512] のテンソルを dim=1 で [32, 30, 512]、[32, 40, 512]、[32, 30, 512] に分割し、逆順で再結合してください。",
            description_en="Split a tensor of shape [32, 100, 512] along dim=1 into [32, 30, 512], [32, 40, 512], [32, 30, 512] and recombine in reverse order.",
            hint_ja="split で分割し、逆順でcat します。",
            hint_en="Use split and cat in reverse order.",
            setup_code="x = torch.randn(32, 100, 512)",
            solution_code="""parts = torch.split(x, [30, 40, 30], dim=1)
result = torch.cat([parts[2], parts[1], parts[0]], dim=1)""",
            tags=["split", "cat", "reorder"],
        ),

        Problem(
            id="stack_010",
            category="stacking_splitting",
            difficulty="advanced",
            title_ja="RGB チャネルの分離と操作",
            title_en="Separate and Manipulate RGB Channels",
            description_ja="形状 [32, 3, 224, 224] の画像から R, G, B チャネルを分離し、R と B を入れ替えて再結合してください。",
            description_en="Separate R, G, B channels from an image of shape [32, 3, 224, 224], swap R and B, and recombine.",
            hint_ja="chunk で分離し、順序を変えて cat します。",
            hint_en="Use chunk to separate, reorder, and cat.",
            setup_code="x = torch.randn(32, 3, 224, 224)",
            solution_code="""channels = torch.chunk(x, chunks=3, dim=1)
result = torch.cat([channels[2], channels[1], channels[0]], dim=1)""",
            tags=["chunk", "cat", "cv", "channels"],
        ),
    ]

    return problems
