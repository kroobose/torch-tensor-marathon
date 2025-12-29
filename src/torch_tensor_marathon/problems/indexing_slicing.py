"""Indexing & Slicing problems - accessing and modifying tensor elements."""

from typing import List
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_indexing_slicing_problems() -> List[Problem]:
    """Get all Indexing & Slicing category problems."""

    problems = [
        Problem(
            id="indexing_basic",
            category="indexing_slicing",
            difficulty="beginner",
            title_ja="Basic Indexing",
            title_en="Basic Indexing",
            cases=[
                ProblemCase(
                    name="Scalar Access",
                    description_ja="テンソル x [10] からインデックス 3 の要素を取得してください。",
                    description_en="Get element at index 3 from x.",
                    hint_ja="x[3] を使用します。",
                    hint_en="Use x[3].",
                    setup_code="x = torch.arange(10)",
                    solution_code="result = x[3]"
                ),
                 ProblemCase(
                    name="Multi-dim Access",
                    description_ja="テンソル x [4, 5] から位置 (1, 2) の要素を取得してください。",
                    description_en="Get element at (1, 2) from x.",
                    hint_ja="x[1, 2] を使用します。",
                    hint_en="Use x[1, 2].",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = x[1, 2]"
                ),
                ProblemCase(
                    name="Negative Index",
                    description_ja="テンソル x の最後の要素を取得してください。",
                    description_en="Get the last element of x.",
                    hint_ja="x[-1] を使用します。",
                    hint_en="Use x[-1].",
                    setup_code="x = torch.arange(10)",
                    solution_code="result = x[-1]"
                ),
                 ProblemCase(
                    name="Negative Multi-dim",
                    description_ja="テンソル x [4, 5] の (0, -2) （0行目、後ろから2列目）を取得してください。",
                    description_en="Get element at (0, -2) from x.",
                    hint_ja="x[0, -2] を使用します。",
                    hint_en="Use x[0, -2].",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = x[0, -2]"
                ),
            ],
            tags=["indexing", "basics"],
        ),

        Problem(
            id="slicing_basic",
            category="indexing_slicing",
            difficulty="beginner",
            title_ja="Slicing Operations",
            title_en="Slicing Operations",
            cases=[
                ProblemCase(
                    name="Range Slice",
                    description_ja="テンソル x [10] のインデックス 2 から 5 (未満) までを取得してください。",
                    description_en="Get elements from index 2 to 5 (exclusive).",
                    hint_ja="x[2:5] を使用します。",
                    hint_en="Use x[2:5].",
                    setup_code="x = torch.arange(10)",
                    solution_code="result = x[2:5]"
                ),
                ProblemCase(
                    name="Step Slice",
                    description_ja="テンソル x [10] のインデックス 1 から 8 まで 2 飛ばしで取得してください。",
                    description_en="Get elements from 1 to 8 with step 2.",
                    hint_ja="x[1:8:2] を使用します。",
                    hint_en="Use x[1:8:2].",
                    setup_code="x = torch.arange(10)",
                    solution_code="result = x[1:8:2]"
                ),
                ProblemCase(
                    name="Multi-dim Slice",
                    description_ja="テンソル x [4, 5] の行 1:3、列 2:4 を取得してください。",
                    description_en="Get rows 1:3 and cols 2:4 from x.",
                    hint_ja="x[1:3, 2:4] を使用します。",
                    hint_en="Use x[1:3, 2:4].",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = x[1:3, 2:4]"
                ),
            ],
            tags=["slicing"],
        ),

        Problem(
            id="indexing_advanced",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Advanced Indexing",
            title_en="Advanced Indexing",
            cases=[
                ProblemCase(
                    name="Integer Array",
                    description_ja="テンソル x からインデックス [1, 4, 7] の要素を取得してください。",
                    description_en="Get elements at indices [1, 4, 7].",
                    hint_ja="x[[1, 4, 7]] を使用します。",
                    hint_en="Use x[[1, 4, 7]].",
                    setup_code="x = torch.arange(10) * 10",
                    solution_code="result = x[torch.tensor([1, 4, 7])]"
                ),
                ProblemCase(
                    name="Select Rows",
                    description_ja="テンソル x [5, 4] から行 [0, 2, 4] を取得してください。",
                    description_en="Get rows [0, 2, 4] from x.",
                    hint_ja="x[[0, 2, 4]] を使用します。",
                    hint_en="Use x[[0, 2, 4]].",
                    setup_code="x = torch.randn(5, 4)",
                    solution_code="result = x[torch.tensor([0, 2, 4])]"
                ),
                ProblemCase(
                    name="Boolean Mask",
                    description_ja="テンソル x から、x > 0 の要素を抽出してください。",
                    description_en="Extract elements where x > 0.",
                    hint_ja="x[x > 0] を使用します。",
                    hint_en="Use x[x > 0].",
                    setup_code="x = torch.arange(10) - 5",
                    solution_code="result = x[x > 0]"
                ),
                ProblemCase(
                    name="Boolean AND",
                    description_ja="テンソル x から、5 < x < 15 の要素を抽出してください。",
                    description_en="Extract elements where 5 < x < 15.",
                    hint_ja="x[(x > 5) & (x < 15)] を使用します。",
                    hint_en="Use x[(x > 5) & (x < 15)].",
                    setup_code="x = torch.arange(20)",
                    solution_code="result = x[(x > 5) & (x < 15)]"
                ),
            ],
            tags=["fancy_indexing", "boolean_indexing"],
        ),

        Problem(
            id="gather_scatter_ops",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Gather & Scatter",
            title_en="Gather & Scatter",
            cases=[
                ProblemCase(
                    name="Gather (dim=1)",
                    description_ja="テンソル x [3, 4] からインデックス [[0,1],[1,2],[2,3]] を dim=1 で gather してください。",
                    description_en="Gather indices along dim 1.",
                    hint_ja="torch.gather(x, 1, indices) を使用します。",
                    hint_en="Use torch.gather(x, 1, indices).",
                    setup_code="""x = torch.randn(3, 4)
indices = torch.tensor([[0, 1], [1, 2], [2, 3]])""",
                    solution_code="result = torch.gather(x, 1, indices)"
                ),
                 ProblemCase(
                    name="Gather (dim=0)",
                    description_ja="テンソル x [4, 3] からインデックス [[0,1,0],[1,2,1]] を dim=0 で gather してください。",
                    description_en="Gather indices along dim 0.",
                    hint_ja="torch.gather(x, 0, indices) を使用します。",
                    hint_en="Use torch.gather(x, 0, indices).",
                    setup_code="""x = torch.randn(4, 3)
indices = torch.tensor([[0, 1, 0], [1, 2, 1]])""",
                    solution_code="result = torch.gather(x, 0, indices)"
                ),
                ProblemCase(
                    name="Scatter Values",
                    description_ja="テンソル x [3, 5] に ones [3, 3] を dim=1, index [[0,1,2]...] で scatter してください。",
                    description_en="Scatter ones into x along dim 1.",
                    hint_ja="x.scatter(1, index, src) を使用します。",
                    hint_en="Use x.scatter(1, index, src).",
                    setup_code="""x = torch.zeros(3, 5)
src = torch.ones(3, 3)
index = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])""",
                    solution_code="result = x.scatter(1, index, src)"
                ),
                ProblemCase(
                    name="Scatter Constant",
                    description_ja="テンソル x に定数 1.0 を指定インデックスに scatter してください。",
                    description_en="Scatter constant 1.0 into x.",
                    hint_ja="x.scatter(1, index, 1.0) を使用します。",
                    hint_en="Use x.scatter(1, index, 1.0).",
                    setup_code="""x = torch.zeros(3, 5)
index = torch.tensor([[0, 2], [1, 3], [2, 4]])""",
                    solution_code="result = x.scatter(1, index, 1.0)"
                ),
            ],
            tags=["gather", "scatter"],
        ),

        Problem(
            id="selection_ops",
            category="indexing_slicing",
            difficulty="advanced",
            title_ja="Selection Operations",
            title_en="Selection Operations",
            cases=[
                ProblemCase(
                    name="Masked Select",
                    description_ja="テンソル x から正の値を masked_select で抽出してください。",
                    description_en="Extract positive values using masked_select.",
                    hint_ja="torch.masked_select(x, x > 0) を使用します。",
                    hint_en="Use torch.masked_select(x, x > 0).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.masked_select(x, x > 0)"
                ),
                 ProblemCase(
                    name="Index Select (Row)",
                    description_ja="テンソル x から行 [0, 2, 4, 6] を index_select で選択してください。",
                    description_en="Select rows using index_select.",
                    hint_ja="torch.index_select(x, 0, indices) を使用します。",
                    hint_en="Use torch.index_select(x, 0, indices).",
                    setup_code="""x = torch.randn(10, 5)
indices = torch.tensor([0, 2, 4, 6])""",
                    solution_code="result = torch.index_select(x, 0, indices)"
                ),
                ProblemCase(
                    name="Index Select (Col)",
                    description_ja="テンソル x から列 [1, 3, 5] を index_select で選択してください。",
                    description_en="Select cols using index_select.",
                    hint_ja="torch.index_select(x, 1, indices) を使用します。",
                    hint_en="Use torch.index_select(x, 1, indices).",
                    setup_code="""x = torch.randn(5, 10)
indices = torch.tensor([1, 3, 5])""",
                    solution_code="result = torch.index_select(x, 1, indices)"
                ),
                ProblemCase(
                    name="Take (Flat)",
                    description_ja="テンソル x を平坦化とみなして、インデックス [0, 5, 10] の値を取得してください (take)。",
                    description_en="Get elements at flat indices [0, 5, 10] using take.",
                    hint_ja="torch.take(x, indices) を使用します。",
                    hint_en="Use torch.take(x, indices).",
                    setup_code="""x = torch.randn(4, 5)
indices = torch.tensor([0, 5, 10])""",
                    solution_code="result = torch.take(x, indices)"
                ),
                ProblemCase(
                    name="Where",
                    description_ja="condition が True なら x, False なら y を選択してください。",
                    description_en="Select x if condition True else y.",
                    hint_ja="torch.where(condition, x, y) を使用します。",
                    hint_en="Use torch.where(condition, x, y).",
                    setup_code="""x = torch.ones(4)
y = torch.zeros(4)
condition = torch.tensor([True, False, True, False])""",
                    solution_code="result = torch.where(condition, x, y)"
                ),
            ],
            tags=["selection", "masked_select", "index_select", "take", "where"],
        ),

        Problem(
            id="writing_ops",
            category="indexing_slicing",
            difficulty="expert",
            title_ja="Writing Operations",
            title_en="Writing Operations",
            cases=[
                ProblemCase(
                    name="Index Put",
                    description_ja="テンソル x の対角成分相当位置に 1.0 を index_put で書き込んでください。",
                    description_en="Write 1.0 to diagonal positions using index_put.",
                    hint_ja="x.index_put(indices, values) を使用します。",
                    hint_en="Use x.index_put(indices, values).",
                    setup_code="""x = torch.zeros(3, 3)
indices = (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
values = torch.tensor([1., 1., 1.])""",
                    solution_code="result = x.index_put(indices, values)"
                ),
                ProblemCase(
                    name="Index Put Accumulate",
                    description_ja="テンソル x の指定位置に値を加算してください (accumulate=True)。",
                    description_en="Accumulate values into x using index_put.",
                    hint_ja="x.index_put(..., accumulate=True) を使用します。",
                    hint_en="Use x.index_put(..., accumulate=True).",
                    setup_code="""x = torch.zeros(5)
indices = (torch.tensor([0, 0, 1]),)
values = torch.tensor([1., 1., 1.])""",
                    solution_code="result = x.index_put(indices, values, accumulate=True)"
                ),
                ProblemCase(
                    name="Index Add",
                    description_ja="テンソル x の dim=0, 指定行に source を加算してください。",
                    description_en="Add source to x at specific rows.",
                    hint_ja="x.index_add(0, index, source) を使用します。",
                    hint_en="Use x.index_add(0, index, source).",
                    setup_code="""x = torch.zeros(5, 3)
index = torch.tensor([0, 2, 4])
source = torch.ones(3, 3)""",
                    solution_code="result = x.index_add(0, index, source)"
                ),
                 ProblemCase(
                    name="Index Copy",
                    description_ja="テンソル x の指定行に source をコピーしてください。",
                    description_en="Copy source into x at specific rows.",
                    hint_ja="x.index_copy(0, index, source) を使用します。",
                    hint_en="Use x.index_copy(0, index, source).",
                    setup_code="""x = torch.zeros(5, 3)
index = torch.tensor([0, 2, 4])
source = torch.ones(3, 3)""",
                    solution_code="result = x.index_copy(0, index, source)"
                ),
                ProblemCase(
                    name="Index Fill",
                    description_ja="テンソル x の指定行を 1.0 で埋めてください。",
                    description_en="Fill specific rows of x with 1.0.",
                    hint_ja="x.index_fill(0, index, 1.0) を使用します。",
                    hint_en="Use x.index_fill(0, index, 1.0).",
                    setup_code="""x = torch.zeros(5, 3)
index = torch.tensor([0, 2, 4])""",
                    solution_code="result = x.index_fill(0, index, 1.0)"
                ),
            ],
            tags=["writing", "index_put", "index_add", "index_fill"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="diagonal_operations",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Diagonal Operations",
            title_en="Diagonal Operations",
            cases=[
                ProblemCase(
                    name="Extract Diagonal",
                    description_ja="行列 x [4, 4] の主対角線を抽出してください。",
                    description_en="Extract main diagonal from x [4, 4].",
                    hint_ja="torch.diagonal(x) または x.diagonal() を使用します。",
                    hint_en="Use torch.diagonal(x) or x.diagonal().",
                    setup_code="x = torch.randn(4, 4)",
                    solution_code="result = torch.diagonal(x)"
                ),
                ProblemCase(
                    name="Diagonal Offset",
                    description_ja="行列 x [5, 5] の対角線オフセット 1 (上対角線) を抽出してください。",
                    description_en="Extract upper diagonal (offset=1) from x.",
                    hint_ja="torch.diagonal(x, offset=1) を使用します。",
                    hint_en="Use torch.diagonal(x, offset=1).",
                    setup_code="x = torch.randn(5, 5)",
                    solution_code="result = torch.diagonal(x, offset=1)"
                ),
                ProblemCase(
                    name="Create Diagonal",
                    description_ja="ベクトル x [4] から対角行列を作成してください。",
                    description_en="Create diagonal matrix from vector x.",
                    hint_ja="torch.diag(x) を使用します。",
                    hint_en="Use torch.diag(x).",
                    setup_code="x = torch.tensor([1., 2., 3., 4.])",
                    solution_code="result = torch.diag(x)"
                ),
            ],
            tags=["diagonal", "diag"],
        ),

        Problem(
            id="triangular_operations",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Triangular Matrix Operations",
            title_en="Triangular Matrix Operations",
            cases=[
                ProblemCase(
                    name="Upper Triangular",
                    description_ja="行列 x [4, 4] の上三角部分を取得してください (対角線含む)。",
                    description_en="Get upper triangular part of x (including diagonal).",
                    hint_ja="torch.triu(x) を使用します。",
                    hint_en="Use torch.triu(x).",
                    setup_code="x = torch.randn(4, 4)",
                    solution_code="result = torch.triu(x)"
                ),
                ProblemCase(
                    name="Lower Triangular",
                    description_ja="行列 x [4, 4] の下三角部分を取得してください (対角線含む)。",
                    description_en="Get lower triangular part of x (including diagonal).",
                    hint_ja="torch.tril(x) を使用します。",
                    hint_en="Use torch.tril(x).",
                    setup_code="x = torch.randn(4, 4)",
                    solution_code="result = torch.tril(x)"
                ),
                ProblemCase(
                    name="Strict Upper",
                    description_ja="行列 x [4, 4] の狭義上三角部分を取得してください (対角線含まず)。",
                    description_en="Get strict upper triangular (excluding diagonal).",
                    hint_ja="torch.triu(x, diagonal=1) を使用します。",
                    hint_en="Use torch.triu(x, diagonal=1).",
                    setup_code="x = torch.randn(4, 4)",
                    solution_code="result = torch.triu(x, diagonal=1)"
                ),
            ],
            tags=["triu", "tril", "triangular"],
        ),

        Problem(
            id="strided_slicing",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Strided Slicing",
            title_en="Strided Slicing",
            cases=[
                ProblemCase(
                    name="Every Other Row",
                    description_ja="テンソル x [10, 5] から偶数行 (0, 2, 4, ...) を取得してください。",
                    description_en="Get even rows from x [10, 5].",
                    hint_ja="x[::2] を使用します。",
                    hint_en="Use x[::2].",
                    setup_code="x = torch.randn(10, 5)",
                    solution_code="result = x[::2]"
                ),
                ProblemCase(
                    name="Every Other Col",
                    description_ja="テンソル x [5, 10] から偶数列を取得してください。",
                    description_en="Get even columns from x [5, 10].",
                    hint_ja="x[:, ::2] を使用します。",
                    hint_en="Use x[:, ::2].",
                    setup_code="x = torch.randn(5, 10)",
                    solution_code="result = x[:, ::2]"
                ),
                ProblemCase(
                    name="Reverse",
                    description_ja="テンソル x [10] を逆順にしてください。",
                    description_en="Reverse tensor x.",
                    hint_ja="torch.flip(x, [0]) または x.flip(0) を使用。※PyTorch では x[::-1] は使用できません。",
                    hint_en="Use torch.flip(x, [0]) or x.flip(0). Note: x[::-1] doesn't work in PyTorch.",
                    setup_code="x = torch.arange(10)",
                    solution_code="result = torch.flip(x, [0])"
                ),
            ],
            tags=["strided", "slicing"],
        ),

        Problem(
            id="multi_index_selection",
            category="indexing_slicing",
            difficulty="advanced",
            title_ja="Multi-Index Selection",
            title_en="Multi-Index Selection",
            cases=[
                ProblemCase(
                    name="Row and Col Indices",
                    description_ja="テンソル x [5, 5] から位置 (row_idx, col_idx) の要素を取得してください。",
                    description_en="Get elements at (row_idx, col_idx) positions.",
                    hint_ja="x[row_idx, col_idx] を使用します。",
                    hint_en="Use x[row_idx, col_idx].",
                    setup_code="""x = torch.randn(5, 5)
row_idx = torch.tensor([0, 1, 2, 3])
col_idx = torch.tensor([4, 3, 2, 1])""",
                    solution_code="result = x[row_idx, col_idx]"
                ),
                ProblemCase(
                    name="Batch Index",
                    description_ja="バッチテンソル x [4, 5, 6] から各バッチの idx[i] 番目を取得してください。",
                    description_en="Get idx[i]-th element from each batch.",
                    hint_ja="x[torch.arange(4), idx] を使用します。",
                    hint_en="Use x[torch.arange(4), idx].",
                    setup_code="""x = torch.randn(4, 5, 6)
idx = torch.tensor([0, 2, 4, 1])""",
                    solution_code="result = x[torch.arange(4), idx]"
                ),
            ],
            tags=["multi_index", "advanced"],
        ),

        Problem(
            id="nonzero_indexing",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Nonzero Indexing",
            title_en="Nonzero Indexing",
            cases=[
                ProblemCase(
                    name="Find Nonzero",
                    description_ja="テンソル x の非ゼロ要素のインデックスを取得してください。",
                    description_en="Get indices of nonzero elements.",
                    hint_ja="torch.nonzero(x) を使用します。",
                    hint_en="Use torch.nonzero(x).",
                    setup_code="x = torch.tensor([0, 1, 0, 2, 0, 3])",
                    solution_code="result = torch.nonzero(x)"
                ),
                ProblemCase(
                    name="Nonzero as Tuple",
                    description_ja="テンソル x の条件 x > 0 を満たすインデックスをタプルで取得してください。",
                    description_en="Get indices where x > 0 as tuple.",
                    hint_ja="torch.nonzero(x > 0, as_tuple=True) または (x > 0).nonzero(as_tuple=True) を使用します。",
                    hint_en="Use torch.nonzero(x > 0, as_tuple=True).",
                    setup_code="x = torch.tensor([-1, 2, -3, 4, -5, 6])",
                    solution_code="result = torch.nonzero(x > 0, as_tuple=True)"
                ),
            ],
            tags=["nonzero"],
        ),

        Problem(
            id="take_along_dim_ops",
            category="indexing_slicing",
            difficulty="advanced",
            title_ja="Take Along Dim",
            title_en="Take Along Dim",
            cases=[
                ProblemCase(
                    name="Argsort Gather",
                    description_ja="テンソル x [3, 4] を行ごとにソートした結果を take_along_dim で取得してください。",
                    description_en="Get sorted values using argsort and take_along_dim.",
                    hint_ja="torch.take_along_dim(x, torch.argsort(x, dim=1), dim=1) を使用します。",
                    hint_en="Use torch.take_along_dim(x, torch.argsort(x, dim=1), dim=1).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.take_along_dim(x, torch.argsort(x, dim=1), dim=1)"
                ),
                ProblemCase(
                    name="Max Index Gather",
                    description_ja="テンソル x [3, 4] の各行の最大値をtake_along_dimで取得してください。",
                    description_en="Get max values per row using take_along_dim.",
                    hint_ja="torch.take_along_dim(x, x.argmax(dim=1, keepdim=True), dim=1) を使用します。",
                    hint_en="Use torch.take_along_dim(x, x.argmax(dim=1, keepdim=True), dim=1).",
                    setup_code="x = torch.randn(3, 4)",
                    solution_code="result = torch.take_along_dim(x, x.argmax(dim=1, keepdim=True), dim=1)"
                ),
            ],
            tags=["take_along_dim", "argsort"],
        ),

        Problem(
            id="clamp_and_index",
            category="indexing_slicing",
            difficulty="intermediate",
            title_ja="Clamp and Index",
            title_en="Clamp and Index",
            cases=[
                ProblemCase(
                    name="Safe Index",
                    description_ja="インデックス idx を 0 から len(x)-1 にクランプしてから x を参照してください。",
                    description_en="Clamp idx to valid range then index x.",
                    hint_ja="x[idx.clamp(0, len(x)-1)] を使用します。",
                    hint_en="Use x[idx.clamp(0, len(x)-1)].",
                    setup_code="""x = torch.arange(10)
idx = torch.tensor([-2, 0, 5, 15])""",
                    solution_code="result = x[idx.clamp(0, len(x)-1)]"
                ),
                ProblemCase(
                    name="Clamp Values",
                    description_ja="テンソル x の値を 0 から 1 の範囲にクランプしてください。",
                    description_en="Clamp x values to [0, 1] range.",
                    hint_ja="x.clamp(0, 1) または torch.clamp(x, 0, 1) を使用します。",
                    hint_en="Use x.clamp(0, 1) or torch.clamp(x, 0, 1).",
                    setup_code="x = torch.randn(10)",
                    solution_code="result = x.clamp(0, 1)"
                ),
            ],
            tags=["clamp", "safe_index"],
        ),

        Problem(
            id="row_col_selection",
            category="indexing_slicing",
            difficulty="beginner",
            title_ja="Row and Column Selection",
            title_en="Row and Column Selection",
            cases=[
                ProblemCase(
                    name="Select Row",
                    description_ja="行列 x [4, 5] の 2 行目を取得してください。",
                    description_en="Get the 2nd row of x [4, 5].",
                    hint_ja="x[2] を使用します。",
                    hint_en="Use x[2].",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = x[2]"
                ),
                ProblemCase(
                    name="Select Column",
                    description_ja="行列 x [4, 5] の 3 列目を取得してください。",
                    description_en="Get the 3rd column of x [4, 5].",
                    hint_ja="x[:, 3] を使用します。",
                    hint_en="Use x[:, 3].",
                    setup_code="x = torch.randn(4, 5)",
                    solution_code="result = x[:, 3]"
                ),
                ProblemCase(
                    name="First N Rows",
                    description_ja="行列 x [10, 5] の最初の 3 行を取得してください。",
                    description_en="Get first 3 rows of x [10, 5].",
                    hint_ja="x[:3] を使用します。",
                    hint_en="Use x[:3].",
                    setup_code="x = torch.randn(10, 5)",
                    solution_code="result = x[:3]"
                ),
                ProblemCase(
                    name="Last N Cols",
                    description_ja="行列 x [5, 10] の最後の 3 列を取得してください。",
                    description_en="Get last 3 columns of x [5, 10].",
                    hint_ja="x[:, -3:] を使用します。",
                    hint_en="Use x[:, -3:].",
                    setup_code="x = torch.randn(5, 10)",
                    solution_code="result = x[:, -3:]"
                ),
            ],
            tags=["row", "column", "selection"],
        ),
    ]

    return problems

