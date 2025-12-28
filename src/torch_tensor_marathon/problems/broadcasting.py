"""Broadcasting & Arithmetic problems - tensor dimension expansion and operations."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_broadcasting_problems() -> List[Problem]:
    """Get all Broadcasting & Arithmetic category problems."""

    problems = [
        # Beginner level
        Problem(
            id="broadcast_001",
            category="broadcasting",
            difficulty="beginner",
            title_ja="スカラーとの演算",
            title_en="Arithmetic with Scalar",
            description_ja="形状 [10, 20] のテンソルに5を加算してください。",
            description_en="Add 5 to a tensor of shape [10, 20].",
            hint_ja="x + 5 でブロードキャストされます。",
            hint_en="x + 5 will broadcast automatically.",
            setup_code="x = torch.randn(10, 20)",
            solution_code="result = x + 5",
            tags=["broadcast", "scalar", "addition"],
        ),

        Problem(
            id="broadcast_002",
            category="broadcasting",
            difficulty="beginner",
            title_ja="次元の追加 - unsqueeze",
            title_en="Add Dimension - unsqueeze",
            description_ja="形状 [10] のテンソルに次元を追加して [10, 1] にしてください。",
            description_en="Add a dimension to a tensor of shape [10] to make it [10, 1].",
            hint_ja="unsqueeze(-1) を使用します。",
            hint_en="Use unsqueeze(-1).",
            setup_code="x = torch.randn(10)",
            solution_code="result = x.unsqueeze(-1)",
            tags=["unsqueeze", "dimension"],
        ),

        Problem(
            id="broadcast_003",
            category="broadcasting",
            difficulty="beginner",
            title_ja="次元の削除 - squeeze",
            title_en="Remove Dimension - squeeze",
            description_ja="形状 [10, 1, 20] のテンソルからサイズ1の次元を削除してください。",
            description_en="Remove dimensions of size 1 from a tensor of shape [10, 1, 20].",
            hint_ja="squeeze() を使用します。",
            hint_en="Use squeeze().",
            setup_code="x = torch.randn(10, 1, 20)",
            solution_code="result = x.squeeze()",
            tags=["squeeze", "dimension"],
        ),

        Problem(
            id="broadcast_004",
            category="broadcasting",
            difficulty="beginner",
            title_ja="ベクトルと行列の加算",
            title_en="Vector and Matrix Addition",
            description_ja="形状 [20] のベクトルを形状 [10, 20] の行列の各行に加算してください。",
            description_en="Add a vector of shape [20] to each row of a matrix of shape [10, 20].",
            hint_ja="x + v でブロードキャストされます。",
            hint_en="x + v will broadcast automatically.",
            setup_code="""x = torch.randn(10, 20)
v = torch.randn(20)""",
            solution_code="result = x + v",
            tags=["broadcast", "vector", "matrix"],
        ),

        # Intermediate level
        Problem(
            id="broadcast_005",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="バッチ正規化スタイルの演算",
            title_en="Batch Normalization Style Operation",
            description_ja="形状 [32, 3, 224, 224] のテンソルから、各チャネルの平均（形状 [3]）を引いてください。結果の形状は [32, 3, 224, 224] です。",
            description_en="Subtract per-channel mean (shape [3]) from a tensor of shape [32, 3, 224, 224]. Result shape is [32, 3, 224, 224].",
            hint_ja="mean を [1, 3, 1, 1] に reshape してブロードキャストします。",
            hint_en="Reshape mean to [1, 3, 1, 1] for broadcasting.",
            setup_code="""x = torch.randn(32, 3, 224, 224)
mean = torch.randn(3)""",
            solution_code="result = x - mean.view(1, 3, 1, 1)",
            tags=["broadcast", "normalization", "cv"],
        ),

        Problem(
            id="broadcast_006",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="expand を使った明示的なブロードキャスト",
            title_en="Explicit Broadcasting with expand",
            description_ja="形状 [1, 5] のテンソルを [10, 5] にブロードキャストしてください（メモリコピーなし）。",
            description_en="Broadcast a tensor of shape [1, 5] to [10, 5] (without memory copy).",
            hint_ja="expand(10, 5) を使用します。",
            hint_en="Use expand(10, 5).",
            setup_code="x = torch.randn(1, 5)",
            solution_code="result = x.expand(10, 5)",
            tags=["expand", "broadcast"],
        ),

        Problem(
            id="broadcast_007",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="repeat を使った実際のコピー",
            title_en="Actual Copy with repeat",
            description_ja="形状 [5] のテンソルを形状 [10, 5] に複製してください（メモリコピーあり）。",
            description_en="Replicate a tensor of shape [5] to [10, 5] (with memory copy).",
            hint_ja="unsqueeze(0).repeat(10, 1) を使用します。",
            hint_en="Use unsqueeze(0).repeat(10, 1).",
            setup_code="x = torch.randn(5)",
            solution_code="result = x.unsqueeze(0).repeat(10, 1)",
            tags=["repeat", "replicate"],
        ),

        Problem(
            id="broadcast_008",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="異なる形状の要素積",
            title_en="Element-wise Product with Different Shapes",
            description_ja="形状 [32, 128, 1] と [1, 1, 512] のテンソルの要素積を計算してください。結果の形状は [32, 128, 512] です。",
            description_en="Compute element-wise product of tensors with shapes [32, 128, 1] and [1, 1, 512]. Result shape is [32, 128, 512].",
            hint_ja="* 演算子を使うと自動的にブロードキャストされます。",
            hint_en="The * operator will broadcast automatically.",
            setup_code="""x = torch.randn(32, 128, 1)
y = torch.randn(1, 1, 512)""",
            solution_code="result = x * y",
            tags=["broadcast", "multiplication"],
        ),

        Problem(
            id="broadcast_009",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="Attention マスクのブロードキャスト",
            title_en="Attention Mask Broadcasting",
            description_ja="形状 [32, 1, 128] のマスクを形状 [32, 8, 128, 128] のアテンションスコアに適用できるように変形してください。",
            description_en="Reshape a mask of shape [32, 1, 128] to broadcast with attention scores of shape [32, 8, 128, 128].",
            hint_ja="unsqueeze を使って [32, 1, 1, 128] にします。",
            hint_en="Use unsqueeze to make it [32, 1, 1, 128].",
            setup_code="mask = torch.randn(32, 1, 128)",
            solution_code="result = mask.unsqueeze(2)",
            tags=["unsqueeze", "attention", "nlp"],
        ),

        Problem(
            id="broadcast_010",
            category="broadcasting",
            difficulty="intermediate",
            title_ja="複数テンソルのブロードキャスト加算",
            title_en="Broadcasting Addition of Multiple Tensors",
            description_ja="形状 [10, 1, 20]、[1, 15, 20]、[10, 15, 1] の3つのテンソルを加算してください。結果の形状は [10, 15, 20] です。",
            description_en="Add three tensors of shapes [10, 1, 20], [1, 15, 20], and [10, 15, 1]. Result shape is [10, 15, 20].",
            hint_ja="a + b + c で自動的にブロードキャストされます。",
            hint_en="a + b + c will broadcast automatically.",
            setup_code="""a = torch.randn(10, 1, 20)
b = torch.randn(1, 15, 20)
c = torch.randn(10, 15, 1)""",
            solution_code="result = a + b + c",
            tags=["broadcast", "multi_tensor"],
        ),

        # Advanced level
        Problem(
            id="broadcast_011",
            category="broadcasting",
            difficulty="advanced",
            title_ja="Position Encoding の加算",
            title_en="Add Position Encoding",
            description_ja="形状 [32, 128, 512] のテンソルに、形状 [128, 512] の位置エンコーディングを加算してください。",
            description_en="Add position encoding of shape [128, 512] to a tensor of shape [32, 128, 512].",
            hint_ja="pos_enc をそのまま加算すると自動的にブロードキャストされます。",
            hint_en="Adding pos_enc directly will broadcast automatically.",
            setup_code="""x = torch.randn(32, 128, 512)
pos_enc = torch.randn(128, 512)""",
            solution_code="result = x + pos_enc",
            tags=["broadcast", "position_encoding", "nlp"],
        ),

        Problem(
            id="broadcast_012",
            category="broadcasting",
            difficulty="advanced",
            title_ja="標準化（平均と標準偏差）",
            title_en="Standardization (Mean and Std)",
            description_ja="形状 [32, 512] のテンソルを、各サンプル（行）ごとに標準化（平均0、標準偏差1）してください。",
            description_en="Standardize a tensor of shape [32, 512] per sample (row) to have mean 0 and std 1.",
            hint_ja="平均と標準偏差を dim=1, keepdim=True で計算します。",
            hint_en="Calculate mean and std with dim=1, keepdim=True.",
            setup_code="x = torch.randn(32, 512)",
            solution_code="""mean = x.mean(dim=1, keepdim=True)
std = x.std(dim=1, keepdim=True)
result = (x - mean) / (std + 1e-8)""",
            tags=["broadcast", "normalization", "statistics"],
        ),
    ]

    return problems
