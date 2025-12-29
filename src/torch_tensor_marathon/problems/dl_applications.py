"""Deep Learning Applications problems - neural network layers and functions."""

from typing import List
import torch.nn.functional as F
from torch_tensor_marathon.problem import Problem, ProblemCase


def get_dl_applications_problems() -> List[Problem]:
    """Get all DL Applications category problems."""

    problems = [
        Problem(
            id="dl_activation_functions",
            category="dl_applications",
            difficulty="beginner",
            title_ja="Activation Functions",
            title_en="Activation Functions",
            cases=[
                ProblemCase(
                    name="ReLU",
                    description_ja="テンソル x に ReLU 活性化関数を適用してください。",
                    description_en="Apply ReLU activation to x.",
                    hint_ja="F.relu(x) を使用します。",
                    hint_en="Use F.relu(x).",
                    setup_code="x = torch.randn(5)",
                    solution_code="result = F.relu(x)"
                ),
                ProblemCase(
                    name="Sigmoid",
                    description_ja="テンソル x に Sigmoid 活性化関数を適用してください。",
                    description_en="Apply Sigmoid activation to x.",
                    hint_ja="torch.sigmoid(x) を使用します。",
                    hint_en="Use torch.sigmoid(x).",
                    setup_code="x = torch.randn(5)",
                    solution_code="result = torch.sigmoid(x)"
                ),
                ProblemCase(
                    name="Softmax",
                    description_ja="テンソル x [10, 5] の最後の次元に対して Softmax を適用してください。",
                    description_en="Apply Softmax to last dim of x.",
                    hint_ja="F.softmax(x, dim=-1) を使用します。",
                    hint_en="Use F.softmax(x, dim=-1).",
                    setup_code="x = torch.randn(10, 5)",
                    solution_code="result = F.softmax(x, dim=-1)"
                ),
            ],
            tags=["relu", "sigmoid", "softmax"],
        ),

        Problem(
            id="dl_vision_layers",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Vision Layers",
            title_en="Vision Layers",
            cases=[
                ProblemCase(
                    name="Conv2d",
                    description_ja="画像 x [B, C, H, W] に重み weight [OutC, C, K, K] で畳み込み(kernel=3, stride=1, padding=1)を適用してください。",
                    description_en="Apply conv2d with k=3, s=1, p=1.",
                    hint_ja="F.conv2d(x, weight, padding=1) を使用します。",
                    hint_en="Use F.conv2d(x, weight, padding=1).",
                    setup_code="""x = torch.randn(2, 3, 32, 32)
weight = torch.randn(16, 3, 3, 3)""",
                    solution_code="result = F.conv2d(x, weight, padding=1)"
                ),
                ProblemCase(
                    name="MaxPool2d",
                    description_ja="画像 x [B, C, H, W] に 2x2 の最大プーリング (stride=2) を適用してください。",
                    description_en="Apply 2x2 max pool with stride 2.",
                    hint_ja="F.max_pool2d(x, 2) を使用します。",
                    hint_en="Use F.max_pool2d(x, 2).",
                    setup_code="x = torch.randn(2, 16, 32, 32)",
                    solution_code="result = F.max_pool2d(x, kernel_size=2, stride=2)"
                ),
                ProblemCase(
                    name="Global Avg Pool",
                    description_ja="画像 x [B, C, H, W] に Global Average Pooling を適用して [B, C, 1, 1] にしてください。",
                    description_en="Apply Global Average Pooling.",
                    hint_ja="F.adaptive_avg_pool2d(x, (1, 1)) を使用します。",
                    hint_en="Use F.adaptive_avg_pool2d(x, (1, 1)).",
                    setup_code="x = torch.randn(2, 64, 8, 8)",
                    solution_code="result = F.adaptive_avg_pool2d(x, (1, 1))"
                ),
                 ProblemCase(
                    name="Pixel Shuffle",
                    description_ja="テンソル x [1, 9, 8, 8] を pixel_shuffle (upscale_factor=3) で [1, 1, 24, 24] にしてください。",
                    description_en="Apply pixel_shuffle with factor 3.",
                    hint_ja="F.pixel_shuffle(x, 3) を使用します。",
                    hint_en="Use F.pixel_shuffle(x, 3).",
                    setup_code="x = torch.randn(1, 9, 8, 8)",
                    solution_code="result = F.pixel_shuffle(x, 3)"
                ),
            ],
            tags=["conv2d", "pooling", "pixel_shuffle"],
        ),

        Problem(
            id="dl_loss_metrics",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Loss & Metrics",
            title_en="Loss & Metrics",
            cases=[
                ProblemCase(
                    name="MSE Loss",
                    description_ja="予測値 pred と正解 target の平均二乗誤差を計算してください。",
                    description_en="Compute MSE loss.",
                    hint_ja="F.mse_loss(pred, target) を使用します。",
                    hint_en="Use F.mse_loss(pred, target).",
                    setup_code="""pred = torch.randn(10)
target = torch.randn(10)""",
                    solution_code="result = F.mse_loss(pred, target)"
                ),
                ProblemCase(
                    name="Cross Entropy",
                    description_ja="ロジット logits [B, C] と正解ラベル target [B] からクロスエントロピー誤差を計算してください。",
                    description_en="Compute Cross Entropy loss.",
                    hint_ja="F.cross_entropy(logits, target) を使用します。",
                    hint_en="Use F.cross_entropy(logits, target).",
                    setup_code="""B, C = 4, 3
logits = torch.randn(B, C)
target = torch.tensor([0, 1, 2, 0])""",
                    solution_code="result = F.cross_entropy(logits, target)"
                ),
                ProblemCase(
                    name="Cosine Similarity",
                    description_ja="ベクトルバッチ x1, x2 [B, D] のコサイン類似度を計算してください (dim=1)。",
                    description_en="Compute cosine similarity along dim 1.",
                    hint_ja="F.cosine_similarity(x1, x2, dim=1) を使用します。",
                    hint_en="Use F.cosine_similarity(x1, x2, dim=1).",
                    setup_code="""x1 = torch.randn(10, 5)
x2 = torch.randn(10, 5)""",
                    solution_code="result = F.cosine_similarity(x1, x2, dim=1)"
                ),
            ],
            tags=["loss", "mse", "cross_entropy", "cosine"],
        ),

        Problem(
            id="dl_linear_ops",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Linear Operations",
            title_en="Linear Operations",
            cases=[
                ProblemCase(
                    name="Linear Layer",
                    description_ja="入力 x [B, I] と重み weight [O, I], バイアス bias [O] を使って線形変換を計算してください。",
                    description_en="Compute linear transformation.",
                    hint_ja="F.linear(x, weight, bias) を使用します。",
                    hint_en="Use F.linear(x, weight, bias).",
                    setup_code="""B, I, O = 4, 10, 5
x = torch.randn(B, I)
weight = torch.randn(O, I)
bias = torch.randn(O)""",
                    solution_code="result = F.linear(x, weight, bias)"
                ),
            ],
            tags=["linear"],
        ),

        Problem(
            id="dl_utilities",
            category="dl_applications",
            difficulty="advanced",
            title_ja="DL Utility Ops",
            title_en="DL Utility Ops",
            cases=[
                ProblemCase(
                    name="Dropout",
                    description_ja="テンソル x に確率 p=0.5 でドロップアウトを適用してください (training=True)。",
                    description_en="Apply dropout with p=0.5 (training=True).",
                    hint_ja="F.dropout(x, p=0.5, training=True) を使用します。",
                    hint_en="Use F.dropout(x, p=0.5, training=True).",
                    setup_code="x = torch.ones(10)",
                    solution_code="result = F.dropout(x, p=0.5, training=True)"
                ),
                ProblemCase(
                    name="Interpolate",
                    description_ja="画像 x [1, 3, 32, 32] を [64, 64] にバイリニア補間でリサイズしてください。",
                    description_en="Resize x to [64, 64] using bilinear interpolation.",
                    hint_ja="F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False) を使用します。",
                    hint_en="Use F.interpolate(..., mode='bilinear', ...).",
                    setup_code="x = torch.randn(1, 3, 32, 32)",
                    solution_code="result = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)"
                ),
                ProblemCase(
                    name="Pad",
                    description_ja="画像 x [1, 1, 4, 4] の周囲に 1px のパディングを追加してください（値は0）。",
                    description_en="Pad x with 1px zero padding.",
                    hint_ja="F.pad(x, (1, 1, 1, 1)) を使用します。",
                    hint_en="Use F.pad(x, (1, 1, 1, 1)).",
                    setup_code="x = torch.randn(1, 1, 4, 4)",
                    solution_code="result = F.pad(x, (1, 1, 1, 1))"
                ),
                ProblemCase(
                    name="Fold (Col2Im)",
                    description_ja="スライド窓展開されたテンソル x [1, 9, 16] を画像 [1, 1, 6, 6] に戻してください (kernel=3)。",
                    description_en="Fold tensor back to image.",
                    hint_ja="F.fold(x, output_size=(6, 6), kernel_size=3) を使用します。",
                    hint_en="Use F.fold(x, output_size=(6, 6), kernel_size=3).",
                    setup_code="x = torch.randn(1, 9, 16)",
                    solution_code="result = F.fold(x, output_size=(6, 6), kernel_size=3)"
                ),
            ],
            tags=["dropout", "interpolate", "pad", "fold"],
        ),

        # === NEW PROBLEMS ===

        Problem(
            id="mha_reshape",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Multi-Head Attention Reshape",
            title_en="Multi-Head Attention Reshape",
            cases=[
                ProblemCase(
                    name="Split Heads",
                    description_ja="テンソル x [B, L, D] を [B, L, num_heads, head_dim] に変換し、[B, num_heads, L, head_dim] に permute してください。",
                    description_en="Split x into heads and permute to [B, H, L, D].",
                    hint_ja="x.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3) を使用します。",
                    hint_en="Use view and permute.",
                    setup_code="""B, L, D = 2, 8, 64
num_heads = 4
head_dim = D // num_heads
x = torch.randn(B, L, D)""",
                    solution_code="result = x.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3)"
                ),
                ProblemCase(
                    name="Merge Heads",
                    description_ja="テンソル x [B, num_heads, L, head_dim] を [B, L, D] に戻してください。",
                    description_en="Merge heads back to [B, L, D].",
                    hint_ja="x.permute(0, 2, 1, 3).reshape(B, L, D) を使用します。",
                    hint_en="Use permute and reshape.",
                    setup_code="""B, L, D = 2, 8, 64
num_heads = 4
head_dim = D // num_heads
x = torch.randn(B, num_heads, L, head_dim)""",
                    solution_code="result = x.permute(0, 2, 1, 3).reshape(B, L, D)"
                ),
            ],
            tags=["mha", "attention", "reshape"],
        ),

        Problem(
            id="layer_norm_reshape",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Layer Normalization",
            title_en="Layer Normalization",
            cases=[
                ProblemCase(
                    name="LayerNorm 1D",
                    description_ja="テンソル x [B, D] に Layer Normalization を適用してください。",
                    description_en="Apply LayerNorm to x [B, D].",
                    hint_ja="F.layer_norm(x, normalized_shape=(D,)) を使用します。",
                    hint_en="Use F.layer_norm(x, (D,)).",
                    setup_code="""B, D = 4, 16
x = torch.randn(B, D)""",
                    solution_code="result = F.layer_norm(x, normalized_shape=(D,))"
                ),
                ProblemCase(
                    name="LayerNorm 2D",
                    description_ja="テンソル x [B, L, D] の最後の次元に Layer Normalization を適用してください。",
                    description_en="Apply LayerNorm to last dim of x [B, L, D].",
                    hint_ja="F.layer_norm(x, normalized_shape=(D,)) を使用します。",
                    hint_en="Use F.layer_norm(x, (D,)).",
                    setup_code="""B, L, D = 4, 8, 16
x = torch.randn(B, L, D)""",
                    solution_code="result = F.layer_norm(x, normalized_shape=(D,))"
                ),
            ],
            tags=["layer_norm", "normalization"],
        ),

        Problem(
            id="position_encoding",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Position Encoding",
            title_en="Position Encoding",
            cases=[
                ProblemCase(
                    name="Simple Position",
                    description_ja="シーケンス長 L のポジション [0, 1, ..., L-1] を [1, L, 1] として作成し、入力 x [B, L, D] にブロードキャストして加算してください。",
                    description_en="Create position tensor [1, L, 1] and add to x [B, L, D].",
                    hint_ja="torch.arange(L).view(1, L, 1) + x を使用します。",
                    hint_en="Use torch.arange(L).view(1, L, 1) + x.",
                    setup_code="""B, L, D = 2, 8, 16
x = torch.randn(B, L, D)""",
                    solution_code="result = torch.arange(L).view(1, L, 1).float() + x"
                ),
                ProblemCase(
                    name="Learned Position",
                    description_ja="位置埋め込み pos_emb [max_len, D] からシーケンス長 L 分を取り出して x [B, L, D] に加算してください。",
                    description_en="Add position embedding to x.",
                    hint_ja="x + pos_emb[:L] を使用します。",
                    hint_en="Use x + pos_emb[:L].",
                    setup_code="""B, L, D = 2, 8, 16
max_len = 64
x = torch.randn(B, L, D)
pos_emb = torch.randn(max_len, D)""",
                    solution_code="result = x + pos_emb[:L]"
                ),
            ],
            tags=["position", "encoding"],
        ),

        Problem(
            id="causal_mask",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Causal Attention Mask",
            title_en="Causal Attention Mask",
            cases=[
                ProblemCase(
                    name="Create Causal Mask",
                    description_ja="シーケンス長 L の Causal Mask (上三角が True) を作成してください。",
                    description_en="Create causal mask of size [L, L].",
                    hint_ja="torch.triu(torch.ones(L, L), diagonal=1).bool() を使用します。",
                    hint_en="Use torch.triu(torch.ones(L, L), diagonal=1).bool().",
                    setup_code="L = 8",
                    solution_code="result = torch.triu(torch.ones(L, L), diagonal=1).bool()"
                ),
                ProblemCase(
                    name="Apply Causal Mask",
                    description_ja="Attention logits [B, H, L, L] に causal_mask を適用して、マスク位置を -inf にしてください。",
                    description_en="Apply causal mask to attention logits.",
                    hint_ja="logits.masked_fill(causal_mask, float('-inf')) を使用します。",
                    hint_en="Use logits.masked_fill(causal_mask, float('-inf')).",
                    setup_code="""B, H, L = 2, 4, 8
logits = torch.randn(B, H, L, L)
causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool()""",
                    solution_code="result = logits.masked_fill(causal_mask, float('-inf'))"
                ),
            ],
            tags=["causal", "mask", "attention"],
        ),

        Problem(
            id="batch_token_masking",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Batch and Token Masking",
            title_en="Batch and Token Masking",
            cases=[
                ProblemCase(
                    name="Padding Mask",
                    description_ja="パディングインデックス pad_idx=0 に対応するマスク [B, L] を作成してください。",
                    description_en="Create padding mask for pad_idx=0.",
                    hint_ja="tokens == pad_idx を使用します。",
                    hint_en="Use tokens == pad_idx.",
                    setup_code="""B, L = 2, 8
pad_idx = 0
tokens = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0]])""",
                    solution_code="result = tokens == pad_idx"
                ),
                ProblemCase(
                    name="Attention Mask Expand",
                    description_ja="パディングマスク [B, L] を Attention マスク [B, 1, 1, L] に拡張してください。",
                    description_en="Expand padding mask for attention.",
                    hint_ja="mask.unsqueeze(1).unsqueeze(2) を使用します。",
                    hint_en="Use mask.unsqueeze(1).unsqueeze(2).",
                    setup_code="""B, L = 2, 8
mask = torch.zeros(B, L).bool()
mask[0, 3:] = True
mask[1, 5:] = True""",
                    solution_code="result = mask.unsqueeze(1).unsqueeze(2)"
                ),
            ],
            tags=["padding", "mask", "attention"],
        ),

        Problem(
            id="embedding_operations",
            category="dl_applications",
            difficulty="beginner",
            title_ja="Embedding Operations",
            title_en="Embedding Operations",
            cases=[
                ProblemCase(
                    name="Embedding Lookup",
                    description_ja="埋め込み行列 embedding [V, D] からトークンインデックス tokens [B, L] の埋め込みを取得してください。",
                    description_en="Get embeddings for tokens from embedding matrix.",
                    hint_ja="F.embedding(tokens, embedding) または embedding[tokens] を使用します。",
                    hint_en="Use F.embedding(tokens, embedding).",
                    setup_code="""V, D = 1000, 64
B, L = 2, 8
embedding = torch.randn(V, D)
tokens = torch.randint(0, V, (B, L))""",
                    solution_code="result = F.embedding(tokens, embedding)"
                ),
                ProblemCase(
                    name="One-hot Embedding",
                    description_ja="トークンインデックス tokens [B, L] を One-hot エンコード [B, L, V] にしてください。",
                    description_en="One-hot encode tokens to [B, L, V].",
                    hint_ja="F.one_hot(tokens, V) を使用します。",
                    hint_en="Use F.one_hot(tokens, V).",
                    setup_code="""V = 10
B, L = 2, 8
tokens = torch.randint(0, V, (B, L))""",
                    solution_code="result = F.one_hot(tokens, V)"
                ),
            ],
            tags=["embedding", "one_hot"],
        ),

        Problem(
            id="normalization_layers",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Normalization Layers",
            title_en="Normalization Layers",
            cases=[
                ProblemCase(
                    name="Batch Norm",
                    description_ja="テンソル x [B, C, H, W] にバッチ正規化を適用してください（running_mean=0, running_var=1, weight=1, bias=0）。",
                    description_en="Apply batch normalization to x.",
                    hint_ja="F.batch_norm(x, running_mean, running_var, ...) を使用します。",
                    hint_en="Use F.batch_norm(...).",
                    setup_code="""B, C, H, W = 2, 16, 8, 8
x = torch.randn(B, C, H, W)
running_mean = torch.zeros(C)
running_var = torch.ones(C)""",
                    solution_code="result = F.batch_norm(x, running_mean, running_var, training=True)"
                ),
                ProblemCase(
                    name="Group Norm",
                    description_ja="テンソル x [B, C, H, W] にグループ正規化を適用してください (num_groups=4)。",
                    description_en="Apply group normalization with 4 groups.",
                    hint_ja="F.group_norm(x, num_groups=4) を使用します。",
                    hint_en="Use F.group_norm(x, num_groups=4).",
                    setup_code="""B, C, H, W = 2, 16, 8, 8
x = torch.randn(B, C, H, W)""",
                    solution_code="result = F.group_norm(x, num_groups=4)"
                ),
            ],
            tags=["batch_norm", "group_norm"],
        ),

        Problem(
            id="attention_patterns",
            category="dl_applications",
            difficulty="expert",
            title_ja="Attention Patterns",
            title_en="Attention Patterns",
            cases=[
                ProblemCase(
                    name="Scaled Dot Product",
                    description_ja="Q, K, V [B, H, L, D] から Scaled Dot-Product Attention を計算してください。",
                    description_en="Compute scaled dot-product attention.",
                    hint_ja="F.scaled_dot_product_attention(Q, K, V) を使用します (PyTorch 2.0+)。",
                    hint_en="Use F.scaled_dot_product_attention (PyTorch 2.0+).",
                    setup_code="""B, H, L, D = 2, 4, 8, 16
Q = torch.randn(B, H, L, D)
K = torch.randn(B, H, L, D)
V = torch.randn(B, H, L, D)""",
                    solution_code="result = F.scaled_dot_product_attention(Q, K, V)"
                ),
                ProblemCase(
                    name="Cross Attention",
                    description_ja="Q [B, H, Lq, D] と K, V [B, H, Lk, D] から Cross Attention を計算してください。",
                    description_en="Compute cross attention Q @ K^T @ V.",
                    hint_ja="F.scaled_dot_product_attention(Q, K, V) を使用します。",
                    hint_en="Use F.scaled_dot_product_attention(Q, K, V).",
                    setup_code="""B, H, Lq, Lk, D = 2, 4, 8, 16, 16
Q = torch.randn(B, H, Lq, D)
K = torch.randn(B, H, Lk, D)
V = torch.randn(B, H, Lk, D)""",
                    solution_code="result = F.scaled_dot_product_attention(Q, K, V)"
                ),
            ],
            tags=["attention", "sdpa"],
        ),

        Problem(
            id="unfold_operations",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Unfold Operations",
            title_en="Unfold Operations",
            cases=[
                ProblemCase(
                    name="Unfold 2D",
                    description_ja="画像 x [B, C, H, W] を kernel_size=3 でパッチに展開してください。",
                    description_en="Unfold x with kernel_size=3.",
                    hint_ja="F.unfold(x, kernel_size=3) を使用します。",
                    hint_en="Use F.unfold(x, kernel_size=3).",
                    setup_code="""B, C, H, W = 1, 3, 8, 8
x = torch.randn(B, C, H, W)""",
                    solution_code="result = F.unfold(x, kernel_size=3)"
                ),
                ProblemCase(
                    name="Unfold with Stride",
                    description_ja="画像 x [B, C, H, W] を kernel_size=3, stride=2 でパッチに展開してください。",
                    description_en="Unfold x with kernel_size=3, stride=2.",
                    hint_ja="F.unfold(x, kernel_size=3, stride=2) を使用します。",
                    hint_en="Use F.unfold(x, kernel_size=3, stride=2).",
                    setup_code="""B, C, H, W = 1, 3, 8, 8
x = torch.randn(B, C, H, W)""",
                    solution_code="result = F.unfold(x, kernel_size=3, stride=2)"
                ),
            ],
            tags=["unfold", "im2col"],
        ),
    ]

    return problems

