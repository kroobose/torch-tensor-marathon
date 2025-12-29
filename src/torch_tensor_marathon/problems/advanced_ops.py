"""Advanced Operations problems - specialized tensor math."""

from typing import List
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_advanced_ops_problems() -> List[Problem]:
    """Get all Advanced Operations category problems."""

    problems = [
        Problem(
            id="block_diagonal_ops",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="Block Diagonal Operations",
            title_en="Block Diagonal Operations",
            cases=[
                ProblemCase(
                    name="Block Diagonal",
                    description_ja="行列 A, B からブロック対角行列を作成してください。",
                    description_en="Create block diagonal matrix from A and B.",
                    hint_ja="torch.block_diag(A, B) を使用します。triu/tril/diag については indexing_slicing カテゴリを参照。",
                    hint_en="Use torch.block_diag(A, B). For triu/tril/diag, see indexing_slicing category.",
                    setup_code="""A = torch.ones(2, 2)
B = torch.zeros(3, 3)""",
                    solution_code="result = torch.block_diag(A, B)"
                ),
                ProblemCase(
                    name="Multi Block Diagonal",
                    description_ja="3つの行列 A, B, C からブロック対角行列を作成してください。",
                    description_en="Create block diagonal from A, B, C.",
                    hint_ja="torch.block_diag(A, B, C) を使用します。",
                    hint_en="Use torch.block_diag(A, B, C).",
                    setup_code="""A = torch.ones(2, 2)
B = torch.randn(3, 3)
C = torch.eye(2)""",
                    solution_code="result = torch.block_diag(A, B, C)"
                ),
            ],
            tags=["block_diag"],
        ),

        Problem(
            id="eye_identity_ops",
            category="advanced_ops",
            difficulty="beginner",
            title_ja="Identity and Special Matrices",
            title_en="Identity and Special Matrices",
            cases=[
                ProblemCase(
                    name="Identity Matrix",
                    description_ja="4x4 の単位行列を作成してください。",
                    description_en="Create a 4x4 identity matrix.",
                    hint_ja="torch.eye(4) を使用します。",
                    hint_en="Use torch.eye(4).",
                    setup_code="",
                    solution_code="result = torch.eye(4)"
                ),
                ProblemCase(
                    name="Ones Like",
                    description_ja="テンソル x と同じ形状・デバイスの全要素1のテンソルを作成してください。",
                    description_en="Create tensor of ones with same shape as x.",
                    hint_ja="torch.ones_like(x) を使用します。",
                    hint_en="Use torch.ones_like(x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.ones_like(x)"
                ),
                ProblemCase(
                    name="Zeros Like",
                    description_ja="テンソル x と同じ形状・デバイスの全要素0のテンソルを作成してください。",
                    description_en="Create tensor of zeros with same shape as x.",
                    hint_ja="torch.zeros_like(x) を使用します。",
                    hint_en="Use torch.zeros_like(x).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.zeros_like(x)"
                ),
            ],
            tags=["eye", "ones", "zeros"],
        ),


        Problem(
            id="sorting_ranking_ops",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="Sorting & Ranking",
            title_en="Sorting & Ranking",
            cases=[
                ProblemCase(
                    name="Top-K Values",
                    description_ja="テンソル x [10] から値が大きい上位3つの要素を取得してください（値のみ）。",
                    description_en="Get top 3 values from x.",
                    hint_ja="torch.topk(x, 3).values を使用します。",
                    hint_en="Use torch.topk(x, 3).values.",
                    setup_code="x = torch.randn(10)",
                    solution_code="result = torch.topk(x, 3).values"
                ),
                ProblemCase(
                    name="Top-K Indices",
                    description_ja="テンソル x [10] から値が大きい上位3つの要素のインデックスを取得してください。",
                    description_en="Get indices of top 3 values from x.",
                    hint_ja="torch.topk(x, 3).indices を使用します。",
                    hint_en="Use torch.topk(x, 3).indices.",
                    setup_code="x = torch.randn(10)",
                    solution_code="result = torch.topk(x, 3).indices"
                ),
                ProblemCase(
                    name="Sort Descending",
                    description_ja="テンソル x [10] を降順にソートしてください（値のみ）。",
                    description_en="Sort x descending.",
                    hint_ja="torch.sort(x, descending=True).values を使用します。",
                    hint_en="Use torch.sort(x, descending=True).values.",
                    setup_code="x = torch.randn(10)",
                    solution_code="result = torch.sort(x, descending=True).values"
                ),
                ProblemCase(
                    name="ArgSort",
                    description_ja="テンソル x [10] を昇順にソートした際のインデックスを取得してください。",
                    description_en="Get argsort of x (ascending).",
                    hint_ja="torch.argsort(x) を使用します。",
                    hint_en="Use torch.argsort(x).",
                    setup_code="x = torch.randn(10)",
                    solution_code="result = torch.argsort(x)"
                ),
            ],
            tags=["sort", "topk", "argsort"],
        ),

        Problem(
            id="unique_counting_ops",
            category="advanced_ops",
            difficulty="advanced",
            title_ja="Unique & Counting",
            title_en="Unique & Counting",
            cases=[
                ProblemCase(
                    name="Unique Values",
                    description_ja="テンソル x [10] に含まれるユニークな値をソートして取得してください。",
                    description_en="Get unique values from x sorted.",
                    hint_ja="torch.unique(x) を使用します。",
                    hint_en="Use torch.unique(x).",
                    setup_code="x = torch.randint(0, 5, (10,))",
                    solution_code="result = torch.unique(x)"
                ),
                ProblemCase(
                    name="Unique Consecutive",
                    description_ja="テンソル x [10] から連続する重複を除いた値を取得してください。",
                    description_en="Get unique consecutive values from x.",
                    hint_ja="torch.unique_consecutive(x) を使用します。",
                    hint_en="Use torch.unique_consecutive(x).",
                    setup_code="x = torch.tensor([1, 1, 2, 2, 3, 1, 1])",
                    solution_code="result = torch.unique_consecutive(x)"
                ),
                ProblemCase(
                    name="Bincount",
                    description_ja="非負整数テンソル x [10] の各値の出現回数をカウントしてください。",
                    description_en="Count occurrences of each value in x.",
                    hint_ja="torch.bincount(x) を使用します。",
                    hint_en="Use torch.bincount(x).",
                    setup_code="x = torch.randint(0, 5, (10,))",
                    solution_code="result = torch.bincount(x)"
                ),
                ProblemCase(
                    name="Histogram",
                    description_ja="テンソル x [100] のヒストグラムを計算してください (bins=10, min=0, max=1)。",
                    description_en="Compute histogram of x (10 bins, 0-1).",
                    hint_ja="torch.histc(x, bins=10, min=0, max=1) を使用します。",
                    hint_en="Use torch.histc(x, bins=10, min=0, max=1).",
                    setup_code="x = torch.rand(100)",
                    solution_code="result = torch.histc(x, bins=10, min=0, max=1)"
                ),
            ],
            tags=["unique", "bincount", "histogram"],
        ),

        Problem(
            id="adv_math_transforms",
            category="advanced_ops",
            difficulty="expert",
            title_ja="Advanced Transforms",
            title_en="Advanced Transforms",
            cases=[
                ProblemCase(
                    name="Lerp",
                    description_ja="start と end の間を weight=0.5 で線形補間してください。",
                    description_en="Linear interpolate between start and end with weight 0.5.",
                    hint_ja="torch.lerp(start, end, 0.5) を使用します。",
                    hint_en="Use torch.lerp(start, end, 0.5).",
                    setup_code="""start = torch.zeros(5)
end = torch.ones(5)""",
                    solution_code="result = torch.lerp(start, end, 0.5)"
                ),
                ProblemCase(
                    name="Renorm",
                    description_ja="行列 x [5, 4] の各行ベクトルの L2 ノルムが maxnorm=1.0 を超えないように正規化してください。",
                    description_en="Renormalize rows of x to have L2 norm <= 1.0.",
                    hint_ja="torch.renorm(x, p=2, dim=0, maxnorm=1.0) を使用します。",
                    hint_en="Use torch.renorm(x, p=2, dim=0, maxnorm=1.0).",
                    setup_code="x = torch.randn(5, 4) * 5",
                    solution_code="result = torch.renorm(x, p=2, dim=0, maxnorm=1.0)"
                ),
                ProblemCase(
                    name="Cartesian Product",
                    description_ja="ベクトル x [3] と y [2] の直積（全組み合わせ）を作成してください ([6, 2])。",
                    description_en="Compute cartesian product of x and y.",
                    hint_ja="torch.cartesian_prod(x, y) を使用します。",
                    hint_en="Use torch.cartesian_prod(x, y).",
                    setup_code="""x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5])""",
                    solution_code="result = torch.cartesian_prod(x, y)"
                ),
            ],
            tags=["lerp", "renorm", "cartesian_prod"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="cumulative_ops",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="Cumulative Operations",
            title_en="Cumulative Operations",
            cases=[
                ProblemCase(
                    name="Cumsum",
                    description_ja="テンソル x [10] の累積和を計算してください。",
                    description_en="Compute cumulative sum of x.",
                    hint_ja="torch.cumsum(x, dim=0) を使用します。",
                    hint_en="Use torch.cumsum(x, dim=0).",
                    setup_code="x = torch.arange(1, 11).float()",
                    solution_code="result = torch.cumsum(x, dim=0)"
                ),
                ProblemCase(
                    name="Cumprod",
                    description_ja="テンソル x [5] の累積積を計算してください。",
                    description_en="Compute cumulative product of x.",
                    hint_ja="torch.cumprod(x, dim=0) を使用します。",
                    hint_en="Use torch.cumprod(x, dim=0).",
                    setup_code="x = torch.arange(1, 6).float()",
                    solution_code="result = torch.cumprod(x, dim=0)"
                ),
                ProblemCase(
                    name="Cumsum 2D",
                    description_ja="テンソル x [3, 4] の dim=1 に沿った累積和を計算してください。",
                    description_en="Compute cumsum along dim 1.",
                    hint_ja="torch.cumsum(x, dim=1) を使用します。",
                    hint_en="Use torch.cumsum(x, dim=1).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.cumsum(x, dim=1)"
                ),
            ],
            tags=["cumsum", "cumprod"],
        ),

        Problem(
            id="roll_shift_ops",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="Roll and Shift Operations",
            title_en="Roll and Shift Operations",
            cases=[
                ProblemCase(
                    name="Roll 1D",
                    description_ja="テンソル x [8] を2つ右にローテーションしてください。",
                    description_en="Roll x by 2 positions to the right.",
                    hint_ja="torch.roll(x, shifts=2) を使用します。",
                    hint_en="Use torch.roll(x, shifts=2).",
                    setup_code="x = torch.arange(8)",
                    solution_code="result = torch.roll(x, shifts=2)"
                ),
                ProblemCase(
                    name="Roll 2D",
                    description_ja="テンソル x [4, 4] を dim=0 で1つ、dim=1 で2つシフトしてください。",
                    description_en="Roll x by (1, 2) on dims (0, 1).",
                    hint_ja="torch.roll(x, shifts=(1, 2), dims=(0, 1)) を使用します。",
                    hint_en="Use torch.roll(x, shifts=(1, 2), dims=(0, 1)).",
                    setup_code="x = torch.arange(16).view(4, 4)",
                    solution_code="result = torch.roll(x, shifts=(1, 2), dims=(0, 1))"
                ),
                ProblemCase(
                    name="Flip",
                    description_ja="テンソル x [4, 4] の dim=1 を反転してください。",
                    description_en="Flip x along dim 1.",
                    hint_ja="torch.flip(x, dims=[1]) を使用します。",
                    hint_en="Use torch.flip(x, dims=[1]).",
                    setup_code="x = torch.arange(16).view(4, 4)",
                    solution_code="result = torch.flip(x, dims=[1])"
                ),
            ],
            tags=["roll", "flip"],
        ),

        Problem(
            id="clamp_normalize_ops",
            category="advanced_ops",
            difficulty="beginner",
            title_ja="Clamp and Normalization",
            title_en="Clamp and Normalization",
            cases=[
                ProblemCase(
                    name="Clamp Range",
                    description_ja="テンソル x の値を [-1, 1] の範囲にクランプしてください。",
                    description_en="Clamp x values to [-1, 1] range.",
                    hint_ja="torch.clamp(x, -1, 1) を使用します。",
                    hint_en="Use torch.clamp(x, -1, 1).",
                    setup_code="x = torch.randn(10) * 3",
                    solution_code="result = torch.clamp(x, -1, 1)"
                ),
                ProblemCase(
                    name="Clamp Min",
                    description_ja="テンソル x の値を 0 以上にクランプしてください (ReLU 相当)。",
                    description_en="Clamp x to min 0 (like ReLU).",
                    hint_ja="torch.clamp(x, min=0) または x.clamp(min=0) を使用します。",
                    hint_en="Use torch.clamp(x, min=0).",
                    setup_code="x = torch.randn(10)",
                    solution_code="result = torch.clamp(x, min=0)"
                ),
                ProblemCase(
                    name="L2 Normalize",
                    description_ja="ベクトル x [10] を L2 正規化してください。",
                    description_en="L2 normalize vector x.",
                    hint_ja="x / x.norm() または F.normalize(x, dim=0) を使用します。",
                    hint_en="Use x / x.norm().",
                    setup_code="x = torch.randn(10)",
                    solution_code="result = x / x.norm()"
                ),
            ],
            tags=["clamp", "normalize"],
        ),

        Problem(
            id="repeat_interleave_ops",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="Repeat Interleave Operations",
            title_en="Repeat Interleave Operations",
            cases=[
                ProblemCase(
                    name="Uniform Repeat",
                    description_ja="テンソル x [4] の各要素を3回繰り返して [12] にしてください。",
                    description_en="Repeat each element of x 3 times.",
                    hint_ja="torch.repeat_interleave(x, 3) を使用します。",
                    hint_en="Use torch.repeat_interleave(x, 3).",
                    setup_code="x = torch.tensor([1, 2, 3, 4])",
                    solution_code="result = torch.repeat_interleave(x, 3)"
                ),
                ProblemCase(
                    name="Variable Repeat",
                    description_ja="テンソル x を repeats で指定された回数ずつ繰り返してください。",
                    description_en="Repeat x elements variable times.",
                    hint_ja="torch.repeat_interleave(x, repeats) を使用します。",
                    hint_en="Use torch.repeat_interleave(x, repeats).",
                    setup_code="""x = torch.tensor([1, 2, 3])
repeats = torch.tensor([1, 2, 3])""",
                    solution_code="result = torch.repeat_interleave(x, repeats)"
                ),
            ],
            tags=["repeat_interleave"],
        ),

        Problem(
            id="meshgrid_ops",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="Meshgrid Operations",
            title_en="Meshgrid Operations",
            cases=[
                ProblemCase(
                    name="2D Grid",
                    description_ja="x 座標 [3] と y 座標 [4] から 2D グリッド (X, Y) を作成してください。",
                    description_en="Create 2D grid from x and y coordinates.",
                    hint_ja="torch.meshgrid(x, y, indexing='ij') を使用します。",
                    hint_en="Use torch.meshgrid(x, y, indexing='ij').",
                    setup_code="""x = torch.arange(3)
y = torch.arange(4)""",
                    solution_code="result = torch.meshgrid(x, y, indexing='ij')"
                ),
                ProblemCase(
                    name="Coordinate Matrix",
                    description_ja="[H, W] のグリッド座標行列を作成してください (torch.stack)。",
                    description_en="Create coordinate matrix [H, W, 2].",
                    hint_ja="meshgrid + stack を使用します。",
                    hint_en="Use meshgrid and stack.",
                    setup_code="""H, W = 4, 5""",
                    solution_code="""grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
result = torch.stack([grid_y, grid_x], dim=-1)"""
                ),
            ],
            tags=["meshgrid"],
        ),

        Problem(
            id="where_select_ops",
            category="advanced_ops",
            difficulty="intermediate",
            title_ja="Where and Select Operations",
            title_en="Where and Select Operations",
            cases=[
                ProblemCase(
                    name="Conditional Select",
                    description_ja="条件 mask に基づいて x または y を選択してください。",
                    description_en="Select x where mask is True, else y.",
                    hint_ja="torch.where(mask, x, y) を使用します。",
                    hint_en="Use torch.where(mask, x, y).",
                    setup_code="""x = torch.ones(5)
y = torch.zeros(5)
mask = torch.tensor([True, False, True, False, True])""",
                    solution_code="result = torch.where(mask, x, y)"
                ),
                ProblemCase(
                    name="Replace NaN",
                    description_ja="テンソル x の NaN 値を 0 に置き換えてください。",
                    description_en="Replace NaN values with 0.",
                    hint_ja="torch.where(torch.isnan(x), 0., x) または x.nan_to_num() を使用します。",
                    hint_en="Use torch.where(torch.isnan(x), 0., x).",
                    setup_code="x = torch.tensor([1., float('nan'), 3., float('nan'), 5.])",
                    solution_code="result = torch.where(torch.isnan(x), torch.zeros_like(x), x)"
                ),
            ],
            tags=["where", "select"],
        ),

        Problem(
            id="linspace_logspace_ops",
            category="advanced_ops",
            difficulty="beginner",
            title_ja="Linspace and Arange",
            title_en="Linspace and Arange",
            cases=[
                ProblemCase(
                    name="Linspace",
                    description_ja="0 から 10 まで等間隔に 11 点を生成してください。",
                    description_en="Generate 11 evenly spaced points from 0 to 10.",
                    hint_ja="torch.linspace(0, 10, 11) を使用します。",
                    hint_en="Use torch.linspace(0, 10, 11).",
                    setup_code="",
                    solution_code="result = torch.linspace(0, 10, 11)"
                ),
                ProblemCase(
                    name="Logspace",
                    description_ja="10^0 から 10^3 まで対数スケールで 4 点を生成してください。",
                    description_en="Generate 4 points from 10^0 to 10^3 on log scale.",
                    hint_ja="torch.logspace(0, 3, 4) を使用します。",
                    hint_en="Use torch.logspace(0, 3, 4).",
                    setup_code="",
                    solution_code="result = torch.logspace(0, 3, 4)"
                ),
            ],
            tags=["linspace", "logspace"],
        ),

        Problem(
            id="masked_operations",
            category="advanced_ops",
            difficulty="advanced",
            title_ja="Masked Operations",
            title_en="Masked Operations",
            cases=[
                ProblemCase(
                    name="Masked Fill",
                    description_ja="テンソル x のマスク位置を -inf で埋めてください。",
                    description_en="Fill masked positions with -inf.",
                    hint_ja="x.masked_fill(mask, float('-inf')) を使用します。",
                    hint_en="Use x.masked_fill(mask, float('-inf')).",
                    setup_code="""x = torch.randn(4, 4)
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()""",
                    solution_code="result = x.masked_fill(mask, float('-inf'))"
                ),
                ProblemCase(
                    name="Masked Select",
                    description_ja="テンソル x からマスクが True の要素を 1D テンソルとして取り出してください。",
                    description_en="Extract elements where mask is True.",
                    hint_ja="torch.masked_select(x, mask) を使用します。",
                    hint_en="Use torch.masked_select(x, mask).",
                    setup_code="""x = torch.randn(4, 4)
mask = x > 0""",
                    solution_code="result = torch.masked_select(x, mask)"
                ),
                ProblemCase(
                    name="Masked Scatter",
                    description_ja="テンソル x のマスク位置に source の値を scatter してください。",
                    description_en="Scatter source into x at masked positions.",
                    hint_ja="x.masked_scatter(mask, source) を使用します。",
                    hint_en="Use x.masked_scatter(mask, source).",
                    setup_code="""x = torch.zeros(5)
mask = torch.tensor([True, False, True, False, True])
source = torch.tensor([1., 2., 3.])""",
                    solution_code="result = x.masked_scatter(mask, source)"
                ),
            ],
            tags=["masked", "fill", "select", "scatter"],
        ),
    ]

    return problems

