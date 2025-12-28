"""Advanced Operations problems - masking, sorting, padding, and more."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_advanced_ops_problems() -> List[Problem]:
    """Get all Advanced Operations category problems."""

    problems = [
        # Intermediate level
        Problem(
            id="advanced_001",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="マスクによる選択",
            title_en="Masked Select",
            description_ja="形状 [10, 20] のテンソルから、マスク（同じ形状、True/False）で True の要素のみを1Dテンソルとして取得してください。",
            description_en="Use a mask (same shape, True/False) to select only True elements from a tensor of shape [10, 20] as a 1D tensor.",
            hint_ja="torch.masked_select(x, mask) を使用します。",
            hint_en="Use torch.masked_select(x, mask).",
            setup_code="""x = torch.randn(10, 20)
mask = x > 0""",
            solution_code="result = torch.masked_select(x, mask)",
            tags=["masked_select", "boolean"],
        ),

        Problem(
            id="advanced_002",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="マスクによる埋め込み",
            title_en="Masked Fill",
            description_ja="形状 [10, 20] のテンソルで、マスクが True の位置を 0 に置き換えてください。",
            description_en="Replace positions where mask is True with 0 in a tensor of shape [10, 20].",
            hint_ja="x.masked_fill(mask, 0) を使用します。",
            hint_en="Use x.masked_fill(mask, 0).",
            setup_code="""x = torch.randn(10, 20)
mask = x < 0""",
            solution_code="result = x.masked_fill(mask, 0)",
            tags=["masked_fill"],
        ),

        Problem(
            id="advanced_003",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="ソートとインデックス",
            title_en="Sort and Indices",
            description_ja="形状 [32, 100] のテンソルを dim=1 でソートし、ソートされた値を返してください。",
            description_en="Sort a tensor of shape [32, 100] along dim=1 and return the sorted values.",
            hint_ja="torch.sort(x, dim=1) を使用します。",
            hint_en="Use torch.sort(x, dim=1).",
            setup_code="x = torch.randn(32, 100)",
            solution_code="""result, _ = torch.sort(x, dim=1)""",
            tags=["sort"],
        ),

        Problem(
            id="advanced_004",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="上三角行列の作成",
            title_en="Create Upper Triangular Matrix",
            description_ja="形状 [10, 10] のテンソルから上三角部分のみを残し、下三角を0にしてください。",
            description_en="Keep only the upper triangular part of a tensor of shape [10, 10] and set the lower triangular part to 0.",
            hint_ja="torch.triu(x) を使用します。",
            hint_en="Use torch.triu(x).",
            setup_code="x = torch.randn(10, 10)",
            solution_code="result = torch.triu(x)",
            tags=["triu", "triangular"],
        ),

        Problem(
            id="advanced_005",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="下三角行列の作成",
            title_en="Create Lower Triangular Matrix",
            description_ja="形状 [10, 10] のテンソルから下三角部分のみを残し、上三角を0にしてください。",
            description_en="Keep only the lower triangular part of a tensor of shape [10, 10] and set the upper triangular part to 0.",
            hint_ja="torch.tril(x) を使用します。",
            hint_en="Use torch.tril(x).",
            setup_code="x = torch.randn(10, 10)",
            solution_code="result = torch.tril(x)",
            tags=["tril", "triangular"],
        ),

        Problem(
            id="advanced_006",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="パディング - 2D",
            title_en="Padding - 2D",
            description_ja="形状 [10, 20] のテンソルに、上下に1、左右に2のゼロパディングを追加してください。結果の形状は [12, 24] です。",
            description_en="Add zero padding to a tensor of shape [10, 20]: 1 on top/bottom, 2 on left/right. Result shape is [12, 24].",
            hint_ja="F.pad(x, (2, 2, 1, 1)) を使用します（左、右、上、下の順）。",
            hint_en="Use F.pad(x, (2, 2, 1, 1)) (left, right, top, bottom order).",
            setup_code="x = torch.randn(10, 20)",
            solution_code="result = F.pad(x, (2, 2, 1, 1))",
            tags=["pad", "padding"],
        ),

        Problem(
            id="advanced_007",
            category="advanced_ops",
            difficulty="advanced",
            title_ja="argsort によるランキング",
            title_en="Ranking with argsort",
            description_ja="形状 [32, 100] のテンソルに対して、各行の要素を降順にソートしたときのインデックスを取得してください。",
            description_en="Get indices that would sort each row of a tensor of shape [32, 100] in descending order.",
            hint_ja="torch.argsort(x, dim=1, descending=True) を使用します。",
            hint_en="Use torch.argsort(x, dim=1, descending=True).",
            setup_code="x = torch.randn(32, 100)",
            solution_code="result = torch.argsort(x, dim=1, descending=True)",
            tags=["argsort", "ranking"],
        ),

        Problem(
            id="advanced_008",
            category="advanced_ops",
            difficulty="advanced",
            title_ja="対角行列の作成",
            title_en="Create Diagonal Matrix",
            description_ja="形状 [10] のベクトルから、それを対角要素とする形状 [10, 10] の対角行列を作成してください。",
            description_en="Create a diagonal matrix of shape [10, 10] from a vector of shape [10] as diagonal elements.",
            hint_ja="torch.diag(v) を使用します。",
            hint_en="Use torch.diag(v).",
            setup_code="v = torch.randn(10)",
            solution_code="result = torch.diag(v)",
            tags=["diag", "diagonal"],
        ),

        Problem(
            id="advanced_009",
            category="advanced_ops",
            difficulty="advanced",
            title_ja="Causal マスクの作成",
            title_en="Create Causal Mask",
            description_ja="形状 [128, 128] の Causal マスク（対角より上が True）を作成してください。",
            description_en="Create a causal mask of shape [128, 128] (True above diagonal).",
            hint_ja="torch.triu(torch.ones(128, 128), diagonal=1).bool() を使用します。",
            hint_en="Use torch.triu(torch.ones(128, 128), diagonal=1).bool().",
            setup_code="",
            solution_code="result = torch.triu(torch.ones(128, 128), diagonal=1).bool()",
            tags=["triu", "mask", "causal", "nlp"],
        ),

        Problem(
            id="advanced_010",
            category="advanced_ops",
            difficulty="advanced",
            title_ja="4Dテンソルのパディング",
            title_en="4D Tensor Padding",
            description_ja="形状 [32, 3, 224, 224] の画像テンソルに、高さと幅の両側に16ピクセルのゼロパディングを追加してください。結果の形状は [32, 3, 256, 256] です。",
            description_en="Add 16 pixels of zero padding on all sides (height and width) to an image tensor of shape [32, 3, 224, 224]. Result shape is [32, 3, 256, 256].",
            hint_ja="F.pad(x, (16, 16, 16, 16)) を使用します。",
            hint_en="Use F.pad(x, (16, 16, 16, 16)).",
            setup_code="x = torch.randn(32, 3, 224, 224)",
            solution_code="result = F.pad(x, (16, 16, 16, 16))",
            tags=["pad", "4d", "cv"],
        ),

        Problem(
            id="advanced_011",
            category="advanced_ops",
            difficulty="expert",
            title_ja="複数の対角線の抽出",
            title_en="Extract Multiple Diagonals",
            description_ja="形状 [100, 100] の行列から、主対角線、1つ上の対角線、1つ下の対角線を抽出し、それらを積み重ねて形状 [3, 99] または [3, 100] のテンソルを作成してください。",
            description_en="Extract the main diagonal, one diagonal above, and one diagonal below from a matrix of shape [100, 100], and stack them into a tensor of shape [3, 99] or [3, 100].",
            hint_ja="torch.diagonal(x, offset=...) を使用し、パディングして stack します。",
            hint_en="Use torch.diagonal(x, offset=...) and pad before stacking.",
            setup_code="x = torch.randn(100, 100)",
            solution_code="""diag_main = torch.diagonal(x, offset=0)
diag_up = torch.diagonal(x, offset=1)
diag_down = torch.diagonal(x, offset=-1)
# Pad to same length
diag_up_padded = F.pad(diag_up, (0, 1))
diag_down_padded = F.pad(diag_down, (0, 1))
result = torch.stack([diag_main, diag_up_padded, diag_down_padded], dim=0)""",
            tags=["diagonal", "stack", "complex"],
        ),

        Problem(
            id="advanced_012",
            category="advanced_ops",
            difficulty="expert",
            title_ja="Reflection パディング",
            title_en="Reflection Padding",
            description_ja="形状 [1, 3, 32, 32] の画像に、高さと幅の両側に4ピクセルの Reflection パディングを追加してください。結果の形状は [1, 3, 40, 40] です。",
            description_en="Add 4 pixels of reflection padding on all sides (height and width) to an image of shape [1, 3, 32, 32]. Result shape is [1, 3, 40, 40].",
            hint_ja="F.pad(x, (4, 4, 4, 4), mode='reflect') を使用します。",
            hint_en="Use F.pad(x, (4, 4, 4, 4), mode='reflect').",
            setup_code="x = torch.randn(1, 3, 32, 32)",
            solution_code="result = F.pad(x, (4, 4, 4, 4), mode='reflect')",
            tags=["pad", "reflection", "cv"],
        ),
    ]

    return problems
