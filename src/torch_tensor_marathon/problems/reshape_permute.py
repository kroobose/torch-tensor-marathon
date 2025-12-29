"""Reshape & Permute problems - tensor shape transformations."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_reshape_permute_problems() -> List[Problem]:
    """Get all Reshape & Permute category problems."""

    problems = [
        # Beginner level - Basic reshape operations
        Problem(
            id="reshape_001",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="1D から 2D への変換",
            title_en="1D to 2D Conversion",
            description_ja="形状 [12] のテンソルを [3, 4] に変換してください。",
            description_en="Convert a tensor of shape [12] to [3, 4].",
            hint_ja="view() または reshape() を使用します。",
            hint_en="Use view() or reshape().",
            setup_code="x = torch.arange(12)",
            solution_code="""result = x.view(3, 4)
# Alternative: result = x.reshape(3, 4)""",
            tags=["reshape", "view", "2d"],
        ),

        Problem(
            id="reshape_002",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="2D から 1D への平坦化",
            title_en="2D to 1D Flattening",
            description_ja="形状 [4, 5] のテンソルを 1次元に平坦化してください。",
            description_en="Flatten a tensor of shape [4, 5] to 1D.",
            hint_ja="view(-1) または flatten() を使用します。",
            hint_en="Use view(-1) or flatten().",
            setup_code="x = torch.randn(4, 5)",
            solution_code="""result = x.view(-1)
# Alternative: result = x.flatten()""",
            tags=["flatten", "view", "1d"],
        ),

        Problem(
            id="reshape_003",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="バッチ次元の追加",
            title_en="Add Batch Dimension",
            description_ja="形状 [3, 224, 224] の画像テンソルにバッチ次元を追加して [1, 3, 224, 224] にしてください。",
            description_en="Add a batch dimension to an image tensor of shape [3, 224, 224] to make it [1, 3, 224, 224].",
            hint_ja="unsqueeze(0) を使用します。",
            hint_en="Use unsqueeze(0).",
            setup_code="x = torch.randn(3, 224, 224)",
            solution_code="""result = x.unsqueeze(0)
# Alternative: result = x[None, ...]""",
            tags=["unsqueeze", "batch", "cv"],
        ),

        Problem(
            id="reshape_004",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="次元の入れ替え（2D）",
            title_en="Transpose 2D",
            description_ja="形状 [3, 5] のテンソルを転置して [5, 3] にしてください。",
            description_en="Transpose a tensor of shape [3, 5] to [5, 3].",
            hint_ja="transpose(0, 1) または .T を使用します。",
            hint_en="Use transpose(0, 1) or .T.",
            setup_code="x = torch.randn(3, 5)",
            solution_code="""result = x.transpose(0, 1)
# Alternative: result = x.T""",
            tags=["transpose", "2d"],
        ),

        # Intermediate level - More complex reshaping
        Problem(
            id="reshape_005",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="チャネル順の変更: NCHW → NHWC",
            title_en="Channel Order Change: NCHW → NHWC",
            description_ja="形状 [32, 3, 224, 224] (バッチ, チャネル, 高さ, 幅) のテンソルを [32, 224, 224, 3] に変換してください。",
            description_en="Convert a tensor of shape [32, 3, 224, 224] (batch, channel, height, width) to [32, 224, 224, 3].",
            hint_ja="permute(0, 2, 3, 1) を使用します。",
            hint_en="Use permute(0, 2, 3, 1).",
            setup_code="x = torch.randn(32, 3, 224, 224)",
            solution_code="result = x.permute(0, 2, 3, 1)",
            tags=["permute", "channel", "cv", "nhwc"],
        ),

        Problem(
            id="reshape_006",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="バッチ行列の転置",
            title_en="Batch Matrix Transpose",
            description_ja="形状 [32, 10, 20] のバッチ行列の最後の2次元を転置してください。",
            description_en="Transpose the last 2 dimensions of a batch matrix of shape [32, 10, 20].",
            hint_ja="transpose(-2, -1) を使用します。",
            hint_en="Use transpose(-2, -1).",
            setup_code="x = torch.randn(32, 10, 20)",
            solution_code="result = x.transpose(-2, -1)",
            tags=["transpose", "batch", "negative_index"],
        ),

        Problem(
            id="reshape_007",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="3D テンソルの整形",
            title_en="3D Tensor Reshaping",
            description_ja="形状 [4, 6, 8] のテンソルを [2, 12, 8] に変換してください。",
            description_en="Reshape a tensor of shape [4, 6, 8] to [2, 12, 8].",
            hint_ja="view() または reshape() を使用します。",
            hint_en="Use view() or reshape().",
            setup_code="x = torch.randn(4, 6, 8)",
            solution_code="""result = x.view(2, 12, 8)
# Alternative: result = x.reshape(2, 12, 8)""",
            tags=["reshape", "3d"],
        ),

        Problem(
            id="reshape_008",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="シーケンス長とバッチの入れ替え",
            title_en="Swap Sequence Length and Batch",
            description_ja="形状 [128, 32, 512] (seq_len, batch, hidden) のテンソルを [32, 128, 512] (batch, seq_len, hidden) に変換してください。",
            description_en="Convert a tensor of shape [128, 32, 512] (seq_len, batch, hidden) to [32, 128, 512] (batch, seq_len, hidden).",
            hint_ja="transpose(0, 1) を使用します。",
            hint_en="Use transpose(0, 1).",
            setup_code="x = torch.randn(128, 32, 512)",
            solution_code="result = x.transpose(0, 1)",
            tags=["transpose", "nlp", "sequence"],
        ),

        Problem(
            id="reshape_009",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="マルチヘッド用のreshape",
            title_en="Reshape for Multi-Head",
            description_ja="形状 [32, 128, 512] のテンソルを8ヘッドに分割: [32, 128, 8, 64] に変換してください。",
            description_en="Split a tensor of shape [32, 128, 512] into 8 heads: convert to [32, 128, 8, 64].",
            hint_ja="view(32, 128, 8, 64) を使用します。",
            hint_en="Use view(32, 128, 8, 64).",
            setup_code="x = torch.randn(32, 128, 512)",
            solution_code="result = x.view(32, 128, 8, 64)",
            tags=["reshape", "attention", "multi_head"],
        ),

        Problem(
            id="reshape_010",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="マルチヘッドの次元順序変更",
            title_en="Multi-Head Dimension Reorder",
            description_ja="形状 [32, 128, 8, 64] のテンソルを [32, 8, 128, 64] (batch, heads, seq_len, head_dim) に変換してください。",
            description_en="Convert a tensor of shape [32, 128, 8, 64] to [32, 8, 128, 64] (batch, heads, seq_len, head_dim).",
            hint_ja="transpose(1, 2) を使用します。",
            hint_en="Use transpose(1, 2).",
            setup_code="x = torch.randn(32, 128, 8, 64)",
            solution_code="result = x.transpose(1, 2)",
            tags=["transpose", "attention", "multi_head"],
        ),

        # Advanced level - Complex permutations and contiguous operations
       Problem(
            id="reshape_011",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="contiguous化が必要なreshape",
            title_en="Reshape Requiring Contiguous",
            description_ja="形状 [32, 8, 128, 64] のテンソル（transpose後）を [32, 8, 128*64] に変換してください。",
            description_en="Reshape a tensor of shape [32, 8, 128, 64] (after transpose) to [32, 8, 128*64].",
            hint_ja="contiguous() を使ってから view() します。",
            hint_en="Use contiguous() before view().",
            setup_code="x = torch.randn(32, 128, 8, 64).transpose(1, 2)",
            solution_code="result = x.contiguous().view(32, 8, 128*64)",
            tags=["contiguous", "view", "attention"],
        ),

        Problem(
            id="reshape_012",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="複雑な次元の並べ替え",
            title_en="Complex Dimension Permutation",
            description_ja="形状 [2, 3, 4, 5, 6] のテンソルを [5, 3, 6, 2, 4] に並べ替えてください。",
            description_en="Permute a tensor of shape [2, 3, 4, 5, 6] to [5, 3, 6, 2, 4].",
            hint_ja="permute(3, 1, 4, 0, 2) を使用します。",
            hint_en="Use permute(3, 1, 4, 0, 2).",
            setup_code="x = torch.randn(2, 3, 4, 5, 6)",
            solution_code="result = x.permute(3, 1, 4, 0, 2)",
            tags=["permute", "5d", "complex"],
        ),

        Problem(
            id="reshape_013",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="パッチ抽出のためのreshape",
            title_en="Reshape for Patch Extraction",
            description_ja="形状 [1, 3, 224, 224] の画像を 16x16 パッチに分割し、[1, 196, 768] に変換してください（ViT用）。",
            description_en="Split an image of shape [1, 3, 224, 224] into 16x16 patches and convert to [1, 196, 768] (for ViT).",
            hint_ja="unfold を使うか、view + permute を組み合わせます。ここでは view を使った解法です。",
            hint_en="Use unfold or combine view + permute. This solution uses view.",
            setup_code="x = torch.randn(1, 3, 224, 224)",
            solution_code="""# Reshape to extract patches: (1, 3, 14, 16, 14, 16)
x_reshaped = x.view(1, 3, 14, 16, 14, 16)
# Permute to (1, 14, 14, 3, 16, 16)
x_permuted = x_reshaped.permute(0, 2, 4, 1, 3, 5)
# Flatten patches: (1, 196, 768)
result = x_permuted.reshape(1, 14*14, 3*16*16)
# Note: Alternatively, use torch.nn.Unfold""",
            tags=["reshape", "permute", "vit", "patches", "cv"],
        ),

        Problem(
            id="reshape_014",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="depthwise conv の出力整形",
            title_en="Depthwise Conv Output Reshaping",
            description_ja="形状 [32, 256, 28, 28] のテンソルをグループ毎に分けて [32, 8, 32, 28, 28] に変換してください。",
            description_en="Split a tensor of shape [32, 256, 28, 28] by groups into [32, 8, 32, 28, 28].",
            hint_ja="view(32, 8, 32, 28, 28) を使用します。",
            hint_en="Use view(32, 8, 32, 28, 28).",
            setup_code="x = torch.randn(32, 256, 28, 28)",
            solution_code="result = x.view(32, 8, 32, 28, 28)",
            tags=["reshape", "cv", "groups"],
        ),

        # Expert level - Very complex or tricky operations
        Problem(
            id="reshape_015",
            category="reshape_permute",
            difficulty="expert",
            title_ja="3D畳み込みの入力変換",
            title_en="3D Convolution Input Conversion",
            description_ja="形状 [16, 10, 3, 112, 112] (batch, frames, channels, H, W) のビデオテンソルを [16, 3, 10, 112, 112] (batch, channels, frames, H, W) に変換してください。",
            description_en="Convert a video tensor of shape [16, 10, 3, 112, 112] (batch, frames, channels, H, W) to [16, 3, 10, 112, 112] (batch, channels, frames, H, W).",
            hint_ja="permute を使って次元を並べ替えます。",
            hint_en="Use permute to reorder dimensions.",
            setup_code="x = torch.randn(16, 10, 3, 112, 112)",
            solution_code="result = x.permute(0, 2, 1, 3, 4)",
            tags=["permute", "3d_conv", "video", "cv"],
        ),
    ]

    return problems
