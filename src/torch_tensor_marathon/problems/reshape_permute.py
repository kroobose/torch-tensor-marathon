"""Reshape & Permute problems - tensor shape transformations."""

from typing import List
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_reshape_permute_problems() -> List[Problem]:
    """Get all Reshape & Permute category problems."""

    problems = [
        Problem(
            id="reshape_basics",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="View vs Reshape Differences",
            title_en="View vs Reshape Differences",
            cases=[
                ProblemCase(
                    name="View (Contiguous)",
                    description_ja="連続メモリのテンソル x [12] を view で (3, 4) に変換してください。",
                    description_en="Convert contiguous tensor x [12] to (3, 4) using view.",
                    hint_ja="x.view(3, 4) を使用。view は連続メモリのテンソルにのみ使用可能です。",
                    hint_en="Use x.view(3, 4). view only works on contiguous tensors.",
                    setup_code="x = torch.arange(12)",
                    solution_code="result = x.view(3, 4)"
                ),
                ProblemCase(
                    name="Reshape (Non-contiguous)",
                    description_ja="転置で非連続になったテンソル x を (6, 2) に変形してください。※view は失敗します。",
                    description_en="Reshape non-contiguous (transposed) tensor x to (6, 2). Note: view would fail here.",
                    hint_ja="x.reshape(6, 2) を使用。reshape は非連続でも動作します（必要ならコピー）。",
                    hint_en="Use x.reshape(6, 2). reshape works on non-contiguous tensors (copies if needed).",
                    setup_code="x = torch.arange(12).view(3, 4).transpose(0, 1)",
                    solution_code="result = x.reshape(6, 2)"
                ),
                ProblemCase(
                    name="Infer Dim (-1)",
                    description_ja="テンソル x [24] を、行数 6、列数自動推論 (-1) で (6, 4) に変換してください。",
                    description_en="Convert x [24] to (6, -1) using -1 for automatic inference.",
                    hint_ja="x.view(6, -1) または x.reshape(6, -1) を使用。-1 は自動的にサイズを推論します。",
                    hint_en="Use x.view(6, -1) or x.reshape(6, -1). -1 infers the size automatically.",
                    setup_code="x = torch.arange(24)",
                    solution_code="result = x.view(6, -1)"
                ),
            ],
            tags=["reshape", "view", "contiguous"],
        ),


        Problem(
            id="flattening",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="Flattening Operations",
            title_en="Flattening Operations",
            cases=[
                ProblemCase(
                    name="Complete Flatten",
                    description_ja="形状 [4, 5] のテンソル x を完全に1次元に平坦化してください。",
                    description_en="Flatten tensor x to 1D.",
                    hint_ja="x.flatten() または x.view(-1) を使用します。",
                    hint_en="Use x.flatten() or x.view(-1).",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = x.flatten()"
                ),
                ProblemCase(
                    name="Batch Flatten",
                    description_ja="形状 [8, 3, 4] のテンソル x のバッチ次元(dim=0)を保持したまま平坦化し、(8, 12)にしてください。",
                    description_en="Flatten x keeping batch dimension (0), resulting in (8, 12).",
                    hint_ja="x.flatten(start_dim=1) を使用します。",
                    hint_en="Use x.flatten(start_dim=1).",
                    setup_code="x = torch.randn(8, 3, 4)",
                    solution_code="result = x.flatten(start_dim=1)"
                ),
                 ProblemCase(
                    name="Range Flatten",
                    description_ja="形状 [2, 3, 4, 5] のテンソル x の dim=1 と dim=2 のみを平坦化し、(2, 12, 5)にしてください。",
                    description_en="Flatten dim 1 and 2 of x, resulting in (2, 12, 5).",
                    hint_ja="x.flatten(start_dim=1, end_dim=2) を使用します。",
                    hint_en="Use x.flatten(start_dim=1, end_dim=2).",
                    setup_code="x = torch.randn(2, 3, 4, 5)",
                    solution_code="result = x.flatten(start_dim=1, end_dim=2)"
                ),
            ],
            tags=["flatten", "reshape"],
        ),

        Problem(
            id="squeeze_unsqueeze",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="Squeeze & Unsqueeze",
            title_en="Squeeze & Unsqueeze",
            cases=[
                 ProblemCase(
                    name="Unsqueeze Front",
                    description_ja="ベクトル x [5] の先頭に次元を追加して [1, 5] にしてください。",
                    description_en="Add dimension at front of x [5] to get [1, 5].",
                    hint_ja="x.unsqueeze(0) を使用します。",
                    hint_en="Use x.unsqueeze(0).",
                    setup_code="x = torch.randn(5)",
                    solution_code="result = x.unsqueeze(0)"
                ),
                ProblemCase(
                    name="Unsqueeze Back",
                    description_ja="ベクトル x [5] の末尾に次元を追加して [5, 1] にしてください。",
                    description_en="Add dimension at back of x [5] to get [5, 1].",
                    hint_ja="x.unsqueeze(-1) を使用します。",
                    hint_en="Use x.unsqueeze(-1).",
                    setup_code="x = torch.randn(5)",
                    solution_code="result = x.unsqueeze(-1)"
                ),
                ProblemCase(
                    name="Squeeze Front",
                    description_ja="テンソル x [1, 5] の先頭のサイズ1の次元を削除してください。",
                    description_en="Remove first size-1 dimension of x [1, 5].",
                    hint_ja="x.squeeze(0) を使用します。",
                    hint_en="Use x.squeeze(0).",
                    setup_code="x = torch.randn(1, 5)",
                    solution_code="result = x.squeeze(0)"
                ),
                 ProblemCase(
                    name="Squeeze Specific",
                    description_ja="テンソル x [5, 1, 4] の dim=1 を削除してください。",
                    description_en="Remove dim=1 from x [5, 1, 4].",
                    hint_ja="x.squeeze(1) を使用します。",
                    hint_en="Use x.squeeze(1).",
                    setup_code="x = torch.randn(5, 1, 4)",
                    solution_code="result = x.squeeze(1)"
                ),
            ],
            tags=["squeeze", "unsqueeze"],
        ),

        Problem(
            id="permute_transpose",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="Permute & Transpose",
            title_en="Permute & Transpose",
            cases=[
                ProblemCase(
                    name="Permute 2D",
                    description_ja="テンソル x [3, 4] の次元を入れ替えて [4, 3] にしてください。",
                    description_en="Permute dimensions of x [3, 4] to [4, 3].",
                    hint_ja="x.permute(1, 0) を使用。2Dでは x.T, x.transpose(0, 1) でも同じ結果になります。",
                    hint_en="Use x.permute(1, 0). For 2D, x.T and x.transpose(0, 1) also work.",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = x.permute(1, 0)"
                ),
                ProblemCase(
                    name="Permute 3D",
                    description_ja="テンソル x [2, 3, 4] の次元を (2, 0, 1) の順序に入れ替え、[4, 2, 3] にしてください。",
                    description_en="Permute dimensions of x [2, 3, 4] to order (2, 0, 1).",
                    hint_ja="x.permute(2, 0, 1) を使用します。",
                    hint_en="Use x.permute(2, 0, 1).",
                    setup_code="x = torch.randn(2, 3, 4)",
                    solution_code="result = x.permute(2, 0, 1)"
                ),
                 ProblemCase(
                    name="CHW to HWC",
                    description_ja="画像テンソル x [3, 224, 224] (CHW) を [224, 224, 3] (HWC) に変換してください。",
                    description_en="Convert image x (CHW) to (HWC).",
                    hint_ja="x.permute(1, 2, 0) を使用します。",
                    hint_en="Use x.permute(1, 2, 0).",
                    setup_code="x = torch.randn(3, 224, 224)",
                    solution_code="result = x.permute(1, 2, 0)"
                ),
                ProblemCase(
                    name="Transpose 2D",
                    description_ja="テンソル x [3, 4] を転置してください。",
                    description_en="Transpose tensor x [3, 4].",
                    hint_ja="x.transpose(0, 1) または x.T を使用します。",
                    hint_en="Use x.transpose(0, 1) or x.T.",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = x.transpose(0, 1)"
                ),
            ],
            tags=["permute", "transpose"],
        ),

        Problem(
            id="reshape_ops_intermediate",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="Dimension Ops (Swap/Move)",
            title_en="Dimension Ops (Swap/Move)",
            cases=[
                ProblemCase(
                    name="Swapdims/Transpose",
                    description_ja="テンソル x [3, 4, 5] の dim=0 と dim=2 を交換してください。",
                    description_en="Swap dim 0 and 2 of x.",
                    hint_ja="torch.swapdims(x, 0, 2) または x.transpose(0, 2) を使用。swapdims は transpose のエイリアスです。",
                    hint_en="Use torch.swapdims(x, 0, 2) or x.transpose(0, 2). swapdims is an alias for transpose.",
                    setup_code="x = torch.randn(3, 4, 5)",
                    solution_code="result = x.transpose(0, 2)"
                ),
                ProblemCase(
                    name="Movedim Single",
                    description_ja="テンソル x [3, 4, 5] の dim=2 を dim=0 に移動してください。",
                    description_en="Move dim 2 to dim 0.",
                    hint_ja="torch.movedim(x, 2, 0) を使用します。",
                    hint_en="Use torch.movedim(x, 2, 0).",
                    setup_code="x = torch.randn(3, 4, 5)",
                    solution_code="result = torch.movedim(x, 2, 0)"
                ),
                 ProblemCase(
                    name="Contiguous",
                    description_ja="転置して不連続になったテンソル x をメモリ上で連続化してください。",
                    description_en="Make transposed tensor x contiguous.",
                    hint_ja="x.contiguous() を使用します。",
                    hint_en="Use x.contiguous().",
                    setup_code="x = torch.randn(3, 4).transpose(0, 1)",
                    solution_code="result = x.contiguous()"
                ),
            ],
            tags=["swapdims", "movedim", "contiguous"],
        ),

        Problem(
            id="reshape_ops_advanced",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="Advanced Reshape Ops",
            title_en="Advanced Reshape Ops",
            cases=[
                ProblemCase(
                    name="Einsum Transpose",
                    description_ja="テンソル x [3, 4] を einsum を使って転置してください。",
                    description_en="Transpose x using einsum.",
                    hint_ja="torch.einsum('ij->ji', x) を使用します。",
                    hint_en="Use torch.einsum('ij->ji', x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.einsum('ij->ji', x)"
                ),
                ProblemCase(
                    name="Repeat (Copy)",
                    description_ja="テンソル x [3, 1] を dim=0で1回、dim=1で4回繰り返して [3, 4] にしてください。",
                    description_en="Repeat x [3, 1] to get [3, 4].",
                    hint_ja="x.repeat(1, 4) を使用。repeat は実際にメモリをコピーします。",
                    hint_en="Use x.repeat(1, 4). repeat actually copies memory.",
                    setup_code="x = torch.randn(3, 1)",
                    solution_code="result = x.repeat(1, 4)"
                ),
                 ProblemCase(
                    name="Expand (View)",
                    description_ja="テンソル x [3, 1] をメモリコピーなしで [3, 4] に拡張してください。",
                    description_en="Expand x [3, 1] to [3, 4] without copying memory.",
                    hint_ja="x.expand(3, 4) を使用。expand はビューを作成（メモリコピーなし）。repeat はコピー。",
                    hint_en="Use x.expand(3, 4). expand creates a view (no copy). repeat copies memory.",
                    setup_code="x = torch.randn(3, 1)",
                    solution_code="result = x.expand(3, 4)"
                ),
                 ProblemCase(
                    name="Unfold",
                    description_ja="テンソル x [10] に対して、dim=0, size=3, step=1 でスライディングウィンドウを適用してください。",
                    description_en="Apply unfold to x [10] with size=3, step=1.",
                    hint_ja="x.unfold(0, 3, 1) を使用します。",
                    hint_en="Use x.unfold(0, 3, 1).",
                    setup_code="x = torch.arange(10)",
                    solution_code="result = x.unfold(0, 3, 1)"
                ),
                ProblemCase(
                    name="Atleast 2D",
                    description_ja="1Dテンソル x [5] を少なくとも2次元のテンソルにしてください。",
                    description_en="Ensure x is at least 2D.",
                    hint_ja="torch.atleast_2d(x) を使用します。",
                    hint_en="Use torch.atleast_2d(x).",
                    setup_code="x = torch.randn(5)",
                    solution_code="result = torch.atleast_2d(x)"
                ),
            ],
            tags=["advanced", "einsum", "repeat", "expand"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="nchw_nhwc_conversion",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="NCHW/NHWC Conversion",
            title_en="NCHW/NHWC Conversion",
            cases=[
                ProblemCase(
                    name="NCHW to NHWC",
                    description_ja="バッチ画像テンソル x [2, 3, 224, 224] (NCHW) を [2, 224, 224, 3] (NHWC) に変換してください。",
                    description_en="Convert batch image x from NCHW to NHWC.",
                    hint_ja="x.permute(0, 2, 3, 1) を使用します。",
                    hint_en="Use x.permute(0, 2, 3, 1).",
                    setup_code="x = torch.randn(2, 3, 224, 224)",
                    solution_code="result = x.permute(0, 2, 3, 1)"
                ),
                ProblemCase(
                    name="NHWC to NCHW",
                    description_ja="バッチ画像テンソル x [2, 224, 224, 3] (NHWC) を [2, 3, 224, 224] (NCHW) に変換してください。",
                    description_en="Convert batch image x from NHWC to NCHW.",
                    hint_ja="x.permute(0, 3, 1, 2) を使用します。",
                    hint_en="Use x.permute(0, 3, 1, 2).",
                    setup_code="x = torch.randn(2, 224, 224, 3)",
                    solution_code="result = x.permute(0, 3, 1, 2)"
                ),
            ],
            tags=["nchw", "nhwc", "image", "permute"],
        ),

        Problem(
            id="view_complex_real",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="Complex Tensor Views",
            title_en="Complex Tensor Views",
            cases=[
                ProblemCase(
                    name="View as Complex",
                    description_ja="形状 [4, 2] の実数テンソル x を複素数テンソル [4] として解釈してください。",
                    description_en="View real tensor x [4, 2] as complex tensor [4].",
                    hint_ja="torch.view_as_complex(x) を使用します。",
                    hint_en="Use torch.view_as_complex(x).",
                    setup_code="x = torch.randn(4, 2)",
                    solution_code="result = torch.view_as_complex(x)"
                ),
                ProblemCase(
                    name="View as Real",
                    description_ja="形状 [4] の複素数テンソル x を実数テンソル [4, 2] として解釈してください。",
                    description_en="View complex tensor x [4] as real tensor [4, 2].",
                    hint_ja="torch.view_as_real(x) を使用します。",
                    hint_en="Use torch.view_as_real(x).",
                    setup_code="x = torch.randn(4, dtype=torch.complex64)",
                    solution_code="result = torch.view_as_real(x)"
                ),
            ],
            tags=["complex", "view", "advanced"],
        ),

        Problem(
            id="unflatten_operations",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="Unflatten Operations",
            title_en="Unflatten Operations",
            cases=[
                ProblemCase(
                    name="Unflatten to 2D",
                    description_ja="形状 [12] のテンソル x を dim=0 で [3, 4] に展開してください。",
                    description_en="Unflatten x [12] at dim 0 to [3, 4].",
                    hint_ja="x.unflatten(0, (3, 4)) を使用します。",
                    hint_en="Use x.unflatten(0, (3, 4)).",
                    setup_code="x = torch.arange(12)",
                    solution_code="result = x.unflatten(0, (3, 4))"
                ),
                ProblemCase(
                    name="Unflatten Middle",
                    description_ja="形状 [2, 12] のテンソル x の dim=1 を [3, 4] に展開して [2, 3, 4] にしてください。",
                    description_en="Unflatten dim 1 of x [2, 12] to [2, 3, 4].",
                    hint_ja="x.unflatten(1, (3, 4)) を使用します。",
                    hint_en="Use x.unflatten(1, (3, 4)).",
                    setup_code="x = torch.randn(2, 12)",
                    solution_code="result = x.unflatten(1, (3, 4))"
                ),
                ProblemCase(
                    name="Unflatten Batch",
                    description_ja="形状 [24, 5] のテンソル x の dim=0 を [4, 6] に展開して [4, 6, 5] にしてください。",
                    description_en="Unflatten dim 0 of x [24, 5] to [4, 6, 5].",
                    hint_ja="x.unflatten(0, (4, 6)) を使用します。",
                    hint_en="Use x.unflatten(0, (4, 6)).",
                    setup_code="x = torch.randn(24, 5)",
                    solution_code="result = x.unflatten(0, (4, 6))"
                ),
            ],
            tags=["unflatten", "reshape"],
        ),

        Problem(
            id="movedim_multiple",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="Moving Multiple Dimensions",
            title_en="Moving Multiple Dimensions",
            cases=[
                ProblemCase(
                    name="Move Two Dims",
                    description_ja="テンソル x [2, 3, 4, 5] の dim 0,1 を dim 2,3 に移動して [4, 5, 2, 3] にしてください。",
                    description_en="Move dims 0,1 to 2,3 for x [2, 3, 4, 5].",
                    hint_ja="torch.movedim(x, (0, 1), (2, 3)) を使用します。",
                    hint_en="Use torch.movedim(x, (0, 1), (2, 3)).",
                    setup_code="x = torch.randn(2, 3, 4, 5)",
                    solution_code="result = torch.movedim(x, (0, 1), (2, 3))"
                ),
                ProblemCase(
                    name="Reverse Dims",
                    description_ja="テンソル x [2, 3, 4] の次元順序を逆にして [4, 3, 2] にしてください。",
                    description_en="Reverse dimension order of x [2, 3, 4].",
                    hint_ja="torch.movedim(x, (0, 1, 2), (2, 1, 0)) を使用します。",
                    hint_en="Use torch.movedim(x, (0, 1, 2), (2, 1, 0)).",
                    setup_code="x = torch.randn(2, 3, 4)",
                    solution_code="result = torch.movedim(x, (0, 1, 2), (2, 1, 0))"
                ),
            ],
            tags=["movedim", "advanced"],
        ),

        Problem(
            id="reshape_chain",
            category="reshape_permute",
            difficulty="advanced",
            title_ja="Chained Reshape Operations",
            title_en="Chained Reshape Operations",
            cases=[
                ProblemCase(
                    name="Flatten then Unflatten",
                    description_ja="テンソル x [2, 3, 4] を平坦化して、再度 [6, 4] に変形してください。",
                    description_en="Flatten x [2, 3, 4] then reshape to [6, 4].",
                    hint_ja="x.flatten().view(6, 4) を使用します。",
                    hint_en="Use x.flatten().view(6, 4).",
                    setup_code="x = torch.randn(2, 3, 4)",
                    solution_code="result = x.flatten().view(6, 4)"
                ),
                ProblemCase(
                    name="Permute then Flatten",
                    description_ja="テンソル x [2, 3, 4] を (1, 2, 0) に並び替えてから平坦化してください。",
                    description_en="Permute x to (1,2,0) then flatten.",
                    hint_ja="x.permute(1, 2, 0).flatten() を使用します。",
                    hint_en="Use x.permute(1, 2, 0).flatten().",
                    setup_code="x = torch.randn(2, 3, 4)",
                    solution_code="result = x.permute(1, 2, 0).flatten()"
                ),
            ],
            tags=["chain", "advanced"],
        ),

        Problem(
            id="batch_reshape",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="Batch-Aware Reshape",
            title_en="Batch-Aware Reshape",
            cases=[
                ProblemCase(
                    name="Reshape Last Dims",
                    description_ja="テンソル x [8, 12] の最後の次元のみを [3, 4] に分割して [8, 3, 4] にしてください。",
                    description_en="Reshape last dim of x [8, 12] to [8, 3, 4].",
                    hint_ja="x.view(8, 3, 4) を使用します。",
                    hint_en="Use x.view(8, 3, 4).",
                    setup_code="x = torch.randn(8, 12)",
                    solution_code="result = x.view(8, 3, 4)"
                ),
                ProblemCase(
                    name="Dynamic Batch",
                    description_ja="任意のバッチサイズのテンソル x [B, 12] を [B, 3, 4] にリシェイプしてください。xの形状は[16, 12]です。",
                    description_en="Reshape x of shape [B, 12] to [B, 3, 4]. x has shape [16, 12].",
                    hint_ja="x.view(-1, 3, 4) または x.view(x.size(0), 3, 4) を使用します。",
                    hint_en="Use x.view(-1, 3, 4) or x.view(x.size(0), 3, 4).",
                    setup_code="x = torch.randn(16, 12)",
                    solution_code="result = x.view(-1, 3, 4)"
                ),
            ],
            tags=["batch", "reshape"],
        ),

        Problem(
            id="tile_operations",
            category="reshape_permute",
            difficulty="intermediate",
            title_ja="Tile Operations",
            title_en="Tile Operations",
            cases=[
                ProblemCase(
                    name="Tile 1D",
                    description_ja="ベクトル x [3] を2回タイルして [6] にしてください。",
                    description_en="Tile vector x [3] to get [6].",
                    hint_ja="torch.tile(x, (2,)) を使用します。",
                    hint_en="Use torch.tile(x, (2,)).",
                    setup_code="x = torch.tensor([1, 2, 3])",
                    solution_code="result = torch.tile(x, (2,))"
                ),
                ProblemCase(
                    name="Tile 2D",
                    description_ja="行列 x [2, 3] を縦2回、横3回タイルして [4, 9] にしてください。",
                    description_en="Tile matrix x [2, 3] to get [4, 9].",
                    hint_ja="torch.tile(x, (2, 3)) を使用します。",
                    hint_en="Use torch.tile(x, (2, 3)).",
                    setup_code="x = torch.randn(2, 3)",
                    solution_code="result = torch.tile(x, (2, 3))"
                ),
            ],
            tags=["tile", "repeat"],
        ),

        Problem(
            id="ravel_patterns",
            category="reshape_permute",
            difficulty="beginner",
            title_ja="Ravel Patterns",
            title_en="Ravel Patterns",
            cases=[
                ProblemCase(
                    name="Ravel 2D",
                    description_ja="行列 x [3, 4] を1次元に変換してください。",
                    description_en="Convert matrix x [3, 4] to 1D.",
                    hint_ja="x.ravel() または torch.ravel(x) を使用します。",
                    hint_en="Use x.ravel() or torch.ravel(x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = x.ravel()"
                ),
                ProblemCase(
                    name="Ravel 3D",
                    description_ja="テンソル x [2, 3, 4] を1次元に変換してください。",
                    description_en="Convert tensor x [2, 3, 4] to 1D.",
                    hint_ja="torch.ravel(x) を使用します。",
                    hint_en="Use torch.ravel(x).",
                    setup_code="x = torch.randn(2, 3, 4)",
                    solution_code="result = torch.ravel(x)"
                ),
            ],
            tags=["ravel", "flatten"],
        ),
    ]

    return problems

