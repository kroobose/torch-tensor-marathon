"""Stacking & Splitting problems - combining and dividing tensors."""

from typing import List
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_stacking_splitting_problems() -> List[Problem]:
    """Get all Stacking & Splitting category problems."""

    problems = [
        Problem(
            id="concatenation",
            category="stacking_splitting",
            difficulty="beginner",
            title_ja="Concatenation",
            title_en="Concatenation",
            cases=[
                ProblemCase(
                    name="cat (dim=0)",
                    description_ja="テンソル x [3, 4] と y [2, 4] を dim=0 で結合してください。",
                    description_en="Concatenate x and y along dim 0.",
                    hint_ja="torch.cat([x, y], dim=0) を使用します。",
                    hint_en="Use torch.cat([x, y], dim=0).",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(2, 4)""",
                    solution_code="result = torch.cat([x, y], dim=0)"
                ),
                ProblemCase(
                    name="cat (dim=1)",
                    description_ja="テンソル x [3, 4] と y [3, 2] を dim=1 で結合してください。",
                    description_en="Concatenate x and y along dim 1.",
                    hint_ja="torch.cat([x, y], dim=1) を使用します。",
                    hint_en="Use torch.cat([x, y], dim=1).",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(3, 2)""",
                    solution_code="result = torch.cat([x, y], dim=1)"
                ),
            ],
            tags=["cat"],
        ),

        Problem(
            id="stacking",
            category="stacking_splitting",
            difficulty="beginner",
            title_ja="Stacking",
            title_en="Stacking",
            cases=[
                ProblemCase(
                    name="stack (dim=0)",
                    description_ja="同じ形状のテンソル x, y [3, 4] を新しい次元 dim=0 で重ねてください。",
                    description_en="Stack x and y along new dim 0.",
                    hint_ja="torch.stack([x, y], dim=0) を使用します。",
                    hint_en="Use torch.stack([x, y], dim=0).",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(3, 4)""",
                    solution_code="result = torch.stack([x, y], dim=0)"
                ),
                ProblemCase(
                    name="stack (dim=1)",
                    description_ja="同じ形状のテンソル x, y [3, 4] を新しい次元 dim=1 で重ねてください。",
                    description_en="Stack x and y along new dim 1.",
                    hint_ja="torch.stack([x, y], dim=1) を使用します。",
                    hint_en="Use torch.stack([x, y], dim=1).",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(3, 4)""",
                    solution_code="result = torch.stack([x, y], dim=1)"
                ),
            ],
            tags=["stack"],
        ),

        Problem(
            id="splitting",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="Splitting Operations",
            title_en="Splitting Operations",
            cases=[
                ProblemCase(
                    name="Chunk",
                    description_ja="テンソル x [6, 4] を dim=0 で 3 つの等しいチャンクに分割してください。",
                    description_en="Chunk x into 3 parts along dim 0.",
                    hint_ja="torch.chunk(x, 3, dim=0) を使用します。",
                    hint_en="Use torch.chunk(x, 3, dim=0).",
                    setup_code="x = torch.randn(6, 4)",
                    solution_code="result = torch.chunk(x, 3, dim=0)"
                ),
                ProblemCase(
                    name="Split (Size)",
                    description_ja="テンソル x [10, 4] を dim=0 でサイズ 2 ずつに分割してください。",
                    description_en="Split x into chunks of size 2 along dim 0.",
                    hint_ja="torch.split(x, 2, dim=0) を使用します。",
                    hint_en="Use torch.split(x, 2, dim=0).",
                    setup_code="x = torch.randn(10, 4)",
                    solution_code="result = torch.split(x, 2, dim=0)"
                ),
                ProblemCase(
                    name="Split (List)",
                    description_ja="テンソル x [10, 4] を dim=0 でサイズ [2, 3, 5] に分割してください。",
                    description_en="Split x into sizes [2, 3, 5] along dim 0.",
                    hint_ja="torch.split(x, [2, 3, 5], dim=0) を使用します。",
                    hint_en="Use torch.split(x, [2, 3, 5], dim=0).",
                    setup_code="x = torch.randn(10, 4)",
                    solution_code="result = torch.split(x, [2, 3, 5], dim=0)"
                ),
                ProblemCase(
                    name="Tensor Split",
                    description_ja="テンソル x [7] を 3 つの部分に分割してください (サイズが割り切れない場合の挙動を確認)。",
                    description_en="Split x [7] into 3 parts.",
                    hint_ja="torch.tensor_split(x, 3) を使用します。",
                    hint_en="Use torch.tensor_split(x, 3).",
                    setup_code="x = torch.arange(7)",
                    solution_code="result = torch.tensor_split(x, 3)"
                ),
            ],
            tags=["split", "chunk"],
        ),

        Problem(
            id="unbind",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="Unbind Operations",
            title_en="Unbind Operations",
            cases=[
                ProblemCase(
                    name="Unbind (dim=0)",
                    description_ja="テンソル x [3, 4] の dim=0 を解いて、(4,) のテンソル3つのタプルにしてください。",
                    description_en="Unbind x along dim 0.",
                    hint_ja="torch.unbind(x, dim=0) を使用します。",
                    hint_en="Use torch.unbind(x, dim=0).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.unbind(x, dim=0)"
                ),
                ProblemCase(
                    name="Unbind (dim=1)",
                    description_ja="テンソル x [3, 4] の dim=1 を解いて、(3,) のテンソル4つのタプルにしてください。",
                    description_en="Unbind x along dim 1.",
                    hint_ja="torch.unbind(x, dim=1) を使用します。",
                    hint_en="Use torch.unbind(x, dim=1).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.unbind(x, dim=1)"
                ),
            ],
            tags=["unbind"],
        ),

        Problem(
            id="stacking_advanced",
            category="stacking_splitting",
            difficulty="advanced",
            title_ja="Advanced Stacking",
            title_en="Advanced Stacking",
            cases=[
                ProblemCase(
                    name="Column Stack",
                    description_ja="1Dテンソルのリスト [a, b, c] (各サイズ [4]) を列として結合し [4, 3] にしてください。",
                    description_en="Stack 1D tensors as columns to get [4, 3].",
                    hint_ja="torch.column_stack([a, b, c]) を使用します。",
                    hint_en="Use torch.column_stack([a, b, c]).",
                    setup_code="""a = torch.randn(4)
b = torch.randn(4)
c = torch.randn(4)""",
                    solution_code="result = torch.column_stack([a, b, c])"
                ),
                ProblemCase(
                    name="Depth Stack (dstack)",
                    description_ja="2Dテンソル x, y [H, W] を深さ方向 (dim=2) に結合して [H, W, 2] にしてください。",
                    description_en="Stack x, y along depth (dim 2).",
                    hint_ja="torch.dstack([x, y]) を使用します。",
                    hint_en="Use torch.dstack([x, y]).",
                    setup_code="""x = torch.randn(10, 10)
y = torch.randn(10, 10)""",
                    solution_code="result = torch.dstack([x, y])"
                ),
                ProblemCase(
                    name="Horizontal Stack (hstack)",
                    description_ja="テンソル x, y [3, 4] を水平方向 (dim=1) に結合して [3, 8] にしてください。",
                    description_en="Stack x, y horizontally (dim 1).",
                    hint_ja="torch.hstack([x, y]) を使用。torch.cat([x, y], dim=1) でも同じ結果。",
                    hint_en="Use torch.hstack([x, y]). torch.cat([x, y], dim=1) also works.",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(3, 4)""",
                    solution_code="result = torch.hstack([x, y])"
                ),
                ProblemCase(
                    name="Vertical Stack (vstack)",
                    description_ja="テンソル x, y [3, 4] を垂直方向 (dim=0) に結合して [6, 4] にしてください。",
                    description_en="Stack x, y vertically (dim 0).",
                    hint_ja="torch.vstack([x, y]) を使用。torch.cat([x, y], dim=0) でも同じ結果。row_stack は vstack のエイリアス。",
                    hint_en="Use torch.vstack([x, y]). torch.cat([x, y], dim=0) also works. row_stack is an alias for vstack.",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(3, 4)""",
                    solution_code="result = torch.vstack([x, y])"
                ),
            ],
            tags=["stack", "advanced"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="hsplit_vsplit_dsplit",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="H/V/D Splitting",
            title_en="H/V/D Splitting",
            cases=[
                ProblemCase(
                    name="hsplit",
                    description_ja="テンソル x [4, 6] を水平方向に 3 等分してください。",
                    description_en="Split x [4, 6] horizontally into 3 parts.",
                    hint_ja="torch.hsplit(x, 3) を使用します。",
                    hint_en="Use torch.hsplit(x, 3).",
                    setup_code="x = torch.randn(4, 6)",
                    solution_code="result = torch.hsplit(x, 3)"
                ),
                ProblemCase(
                    name="vsplit",
                    description_ja="テンソル x [6, 4] を垂直方向に 3 等分してください。",
                    description_en="Split x [6, 4] vertically into 3 parts.",
                    hint_ja="torch.vsplit(x, 3) を使用します。",
                    hint_en="Use torch.vsplit(x, 3).",
                    setup_code="x = torch.randn(6, 4)",
                    solution_code="result = torch.vsplit(x, 3)"
                ),
                ProblemCase(
                    name="dsplit",
                    description_ja="テンソル x [4, 4, 6] を深さ方向に 3 等分してください。",
                    description_en="Split x [4, 4, 6] along depth into 3 parts.",
                    hint_ja="torch.dsplit(x, 3) を使用します。",
                    hint_en="Use torch.dsplit(x, 3).",
                    setup_code="x = torch.randn(4, 4, 6)",
                    solution_code="result = torch.dsplit(x, 3)"
                ),
            ],
            tags=["hsplit", "vsplit", "dsplit"],
        ),

        Problem(
            id="tensor_split_uneven",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="Uneven Tensor Split",
            title_en="Uneven Tensor Split",
            cases=[
                ProblemCase(
                    name="Indices Split",
                    description_ja="テンソル x [10] を位置 [2, 5] で分割してください ([0:2], [2:5], [5:])。",
                    description_en="Split x at indices [2, 5].",
                    hint_ja="torch.tensor_split(x, [2, 5]) を使用します。",
                    hint_en="Use torch.tensor_split(x, [2, 5]).",
                    setup_code="x = torch.arange(10)",
                    solution_code="result = torch.tensor_split(x, [2, 5])"
                ),
                ProblemCase(
                    name="Uneven N Split",
                    description_ja="テンソル x [11] を 4 つの部分に分割してください。",
                    description_en="Split x [11] into 4 parts.",
                    hint_ja="torch.tensor_split(x, 4) を使用します。",
                    hint_en="Use torch.tensor_split(x, 4).",
                    setup_code="x = torch.arange(11)",
                    solution_code="result = torch.tensor_split(x, 4)"
                ),
            ],
            tags=["tensor_split", "uneven"],
        ),

        # Note: row_stack は vstack のエイリアスのため削除。vstack を使用してください。

        Problem(
            id="chunk_process_reassemble",
            category="stacking_splitting",
            difficulty="advanced",
            title_ja="Chunk, Process, Reassemble",
            title_en="Chunk, Process, Reassemble",
            cases=[
                ProblemCase(
                    name="Chunk Add Reassemble",
                    description_ja="テンソル x [9, 4] を 3 つに分割し、各チャンクに 1.0 を足してから cat で結合してください。",
                    description_en="Chunk x, add 1.0 to each, then cat back.",
                    hint_ja="torch.cat([c + 1.0 for c in torch.chunk(x, 3)]) を使用します。",
                    hint_en="Use torch.cat([c + 1.0 for c in torch.chunk(x, 3)]).",
                    setup_code="x = torch.randn(9, 4)",
                    solution_code="result = torch.cat([c + 1.0 for c in torch.chunk(x, 3)])"
                ),
                ProblemCase(
                    name="Split Transform Stack",
                    description_ja="テンソル x [8, 4] を 4 つに分割し、それぞれの平均を計算してから stack してください。",
                    description_en="Split x into 4, compute mean of each, then stack.",
                    hint_ja="torch.stack([c.mean() for c in torch.chunk(x, 4)]) を使用します。",
                    hint_en="Use torch.stack([c.mean() for c in torch.chunk(x, 4)]).",
                    setup_code="x = torch.randn(8, 4)",
                    solution_code="result = torch.stack([c.mean() for c in torch.chunk(x, 4)])"
                ),
            ],
            tags=["chunk", "process", "reassemble"],
        ),

        Problem(
            id="split_along_batch",
            category="stacking_splitting",
            difficulty="intermediate",
            title_ja="Split Along Batch",
            title_en="Split Along Batch",
            cases=[
                ProblemCase(
                    name="Mini-batch Split",
                    description_ja="バッチテンソル x [32, 10] をミニバッチサイズ 8 で分割してください。",
                    description_en="Split batch x [32, 10] into mini-batches of size 8.",
                    hint_ja="torch.split(x, 8, dim=0) を使用します。",
                    hint_en="Use torch.split(x, 8, dim=0).",
                    setup_code="x = torch.randn(32, 10)",
                    solution_code="result = torch.split(x, 8, dim=0)"
                ),
                ProblemCase(
                    name="Per-Sample Split",
                    description_ja="バッチテンソル x [4, 3, 32, 32] を各サンプルに分割してください。",
                    description_en="Split batch x into individual samples.",
                    hint_ja="torch.unbind(x, dim=0) を使用します。",
                    hint_en="Use torch.unbind(x, dim=0).",
                    setup_code="x = torch.randn(4, 3, 32, 32)",
                    solution_code="result = torch.unbind(x, dim=0)"
                ),
            ],
            tags=["batch", "split"],
        ),

        Problem(
            id="interleaved_stacking",
            category="stacking_splitting",
            difficulty="advanced",
            title_ja="Interleaved Stacking",
            title_en="Interleaved Stacking",
            cases=[
                ProblemCase(
                    name="Interleave 2 Tensors",
                    description_ja="テンソル x, y [4] を交互に並べて [8] にしてください (x0, y0, x1, y1, ...)。",
                    description_en="Interleave x and y to get [8].",
                    hint_ja="torch.stack([x, y], dim=1).flatten() を使用します。",
                    hint_en="Use torch.stack([x, y], dim=1).flatten().",
                    setup_code="""x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([10, 20, 30, 40])""",
                    solution_code="result = torch.stack([x, y], dim=1).flatten()"
                ),
                ProblemCase(
                    name="Interleave Rows",
                    description_ja="行列 x, y [3, 4] の行を交互に並べて [6, 4] にしてください。",
                    description_en="Interleave rows of x and y.",
                    hint_ja="torch.stack([x, y], dim=1).view(-1, 4) を使用します。",
                    hint_en="Use torch.stack([x, y], dim=1).view(-1, 4).",
                    setup_code="""x = torch.randn(3, 4)
y = torch.randn(3, 4)""",
                    solution_code="result = torch.stack([x, y], dim=1).view(-1, 4)"
                ),
            ],
            tags=["interleave", "stack"],
        ),

        Problem(
            id="unbind_transform_rebind",
            category="stacking_splitting",
            difficulty="advanced",
            title_ja="Unbind, Transform, Rebind",
            title_en="Unbind, Transform, Rebind",
            cases=[
                ProblemCase(
                    name="Unbind Scale Stack",
                    description_ja="テンソル x [3, 4] を unbind し、各行を (i+1) 倍してから stack してください。",
                    description_en="Unbind x, scale each row by (i+1), then stack.",
                    hint_ja="torch.stack([r * (i+1) for i, r in enumerate(torch.unbind(x))]) を使用します。",
                    hint_en="Use torch.stack([r * (i+1) for i, r in enumerate(torch.unbind(x))]).",
                    setup_code="x = torch.ones(3, 4)",
                    solution_code="result = torch.stack([r * (i+1) for i, r in enumerate(torch.unbind(x))])"
                ),
                ProblemCase(
                    name="Unbind Reverse Stack",
                    description_ja="テンソル x [4, 3] を unbind し、逆順に stack してください。",
                    description_en="Unbind x, reverse, then stack.",
                    hint_ja="torch.stack(list(reversed(torch.unbind(x)))) を使用します。",
                    hint_en="Use torch.stack(list(reversed(torch.unbind(x)))).",
                    setup_code="x = torch.arange(12).view(4, 3)",
                    solution_code="result = torch.stack(list(reversed(torch.unbind(x))))"
                ),
            ],
            tags=["unbind", "transform", "stack"],
        ),
    ]

    return problems

