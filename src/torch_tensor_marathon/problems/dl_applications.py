"""Deep Learning Applications problems - practical scenarios from real DL tasks."""

from typing import List
from torch_tensor_marathon.problem import Problem


def get_dl_applications_problems() -> List[Problem]:
    """Get all Deep Learning Applications category problems."""

    problems = [
        # Intermediate level - Practical DL scenarios
        Problem(
            id="dl_001",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Vision Transformer のパッチ埋め込み",
            title_en="Vision Transformer Patch Embedding",
            description_ja="形状 [1, 3, 224, 224] の画像を 16x16 パッチに分割し、[1, 196, 768] に変換してください（パッチサイズ16、196=14*14パッチ、768=3*16*16）。",
            description_en="Split an image of shape [1, 3, 224, 224] into 16x16 patches and reshape to [1, 196, 768] (patch size 16, 196=14*14 patches, 768=3*16*16).",
            hint_ja="unfold または view + permute + reshape を使用します。",
            hint_en="Use unfold or view + permute + reshape.",
            setup_code="x = torch.randn(1, 3, 224, 224)",
            solution_code="""# Using unfold
patches = x.unfold(2, 16, 16).unfold(3, 16, 16)  # [1, 3, 14, 14, 16, 16]
result = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 196, 768)""",
            tags=["vit", "patches", "cv", "unfold"],
        ),

        Problem(
            id="dl_002",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Positional Encoding の作成",
            title_en="Create Positional Encoding",
            description_ja="形状 [128, 512] の位置エンコーディングを作成してください。位置0~127、次元0~511で、偶数次元は sin、奇数次元は cos を使用します（簡易版：ランダムで OK）。",
            description_en="Create a positional encoding of shape [128, 512]. For positions 0-127 and dimensions 0-511 (simplified: random is OK for this exercise).",
            hint_ja="実際は sin/cos ですが、ここでは形状が正しければ OK です。",
            hint_en="Actual implementation uses sin/cos, but correct shape is sufficient here.",
            setup_code="""position = torch.arange(128).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 512, 2) * -(torch.log(torch.tensor(10000.0)) / 512))""",
            solution_code="""result = torch.zeros(128, 512)
result[:, 0::2] = torch.sin(position * div_term)
result[:, 1::2] = torch.cos(position * div_term)""",
            tags=["positional_encoding", "nlp", "transformer"],
        ),

        Problem(
            id="dl_003",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Attention マスクの作成（パディング用）",
            title_en="Create Attention Mask (for Padding)",
            description_ja="形状 [32] のシーケンス長テンソル（各値は実際の長さ、最大128）から、形状 [32, 128] のパディングマスクを作成してください（実際の長さまでTrue、それ以降False）。",
            description_en="Create a padding mask of shape [32, 128] from a sequence length tensor of shape [32] (each value is actual length, max 128). True up to actual length, False afterwards.",
            hint_ja="arange と比較を組み合わせます。",
            hint_en="Combine arange and comparison.",
            setup_code="seq_lengths = torch.randint(50, 128, (32,))",
            solution_code="""positions = torch.arange(128).unsqueeze(0)  # [1, 128]
result = positions < seq_lengths.unsqueeze(1)  # [32, 128]""",
            tags=["mask", "padding", "nlp"],
        ),

        Problem(
            id="dl_004",
            category="dl_applications",
            difficulty="intermediate",
            title_ja="Mixup データ拡張",
            title_en="Mixup Data Augmentation",
            description_ja="形状 [32, 3, 224, 224] の画像バッチに対して、lambda=0.3 で Mixup を適用してください（各画像とランダムにシャッフルした画像を混合）。",
            description_en="Apply Mixup with lambda=0.3 to a batch of images of shape [32, 3, 224, 224] (mix each image with a randomly shuffled image).",
            hint_ja="torch.randperm でインデックスをシャッフルし、重み付き和を取ります。",
            hint_en="Use torch.randperm to shuffle indices and take weighted sum.",
            setup_code="""x = torch.randn(32, 3, 224, 224)
lam = 0.3""",
            solution_code="""indices = torch.randperm(32)
result = lam * x + (1 - lam) * x[indices]""",
            tags=["mixup", "augmentation", "cv"],
        ),

        Problem(
            id="dl_005",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Focal Loss の重み計算",
            title_en="Focal Loss Weight Computation",
            description_ja="形状 [32, 10] のロジットと形状 [32] のラベルから、Focal Loss 用の重み (1 - p_t)^gamma を計算してください（gamma=2.0）。結果の形状は [32] です。",
            description_en="Compute Focal Loss weights (1 - p_t)^gamma from logits of shape [32, 10] and labels of shape [32] (gamma=2.0). Result shape is [32].",
            hint_ja="softmax で確率を計算し、正解クラスの確率を gather します。",
            hint_en="Compute probabilities with softmax and gather correct class probabilities.",
            setup_code="""logits = torch.randn(32, 10)
labels = torch.randint(0, 10, (32,))
gamma = 2.0""",
            solution_code="""probs = F.softmax(logits, dim=1)
p_t = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
result = (1 - p_t) ** gamma""",
            tags=["focal_loss", "ml", "weights"],
        ),

        Problem(
            id="dl_006",
            category="dl_applications",
            difficulty="advanced",
            title_ja="CutMix のマスク作成",
            title_en="Create CutMix Mask",
            description_ja="224x224 の画像用に、ランダムな位置に 56x56 の矩形マスクを作成してください。結果の形状は [224, 224] で、矩形内が True、外が False です。",
            description_en="Create a 56x56 rectangular mask at a random position for a 224x224 image. Result shape is [224, 224], True inside rectangle, False outside.",
            hint_ja="ランダムな左上座標を選び、スライスで True を設定します。",
            hint_en="Choose random top-left coordinates and set True using slices.",
            setup_code="""torch.manual_seed(42)
x_start = torch.randint(0, 224 - 56, (1,)).item()
y_start = torch.randint(0, 224 - 56, (1,)).item()""",
            solution_code="""result = torch.zeros(224, 224, dtype=torch.bool)
result[y_start:y_start+56, x_start:x_start+56] = True""",
            tags=["cutmix", "mask", "cv", "augmentation"],
        ),

        Problem(
            id="dl_007",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Gradient Clipping の準備",
            title_en="Prepare for Gradient Clipping",
            description_ja="形状 [100, 512] のグラディエントテンソルのL2ノルムを計算し、それが10より大きい場合は10にクリップしたスケーリング係数を返してください。",
            description_en="Compute the L2 norm of a gradient tensor of shape [100, 512], and if it's greater than 10, return the scaling factor to clip it to 10.",
            hint_ja="torch.norm() でノルムを計算し、torch.where で条件分岐します。",
            hint_en="Use torch.norm() to compute norm and torch.where for conditional.",
            setup_code="""grad = torch.randn(100, 512)
max_norm = 10.0""",
            solution_code="""total_norm = torch.norm(grad)
result = torch.where(total_norm > max_norm, max_norm / total_norm, torch.tensor(1.0))""",
            tags=["gradient", "clipping", "training"],
        ),

        Problem(
            id="dl_008",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Multi-task Loss のマスク適用",
            title_en="Apply Multi-task Loss Mask",
            description_ja="形状 [32, 3] の3タスクのロスと、形状 [32, 3] のタスク有効性マスク（0または1）があります。各サンプルで有効なタスクのロスのみを平均してください。結果の形状は [32] です。",
            description_en="Given losses of shape [32, 3] for 3 tasks and a task validity mask of shape [32, 3] (0 or 1), average only valid task losses for each sample. Result shape is [32].",
            hint_ja="マスクで無効なロスを0にし、有効なタスク数で割ります。",
            hint_en="Mask invalid losses to 0 and divide by number of valid tasks.",
            setup_code="""losses = torch.randn(32, 3).abs()
mask = torch.randint(0, 2, (32, 3)).float()""",
            solution_code="""masked_losses = losses * mask
num_valid = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
result = masked_losses.sum(dim=1) / num_valid""",
            tags=["multi_task", "loss", "masking"],
        ),

        Problem(
            id="dl_009",
            category="dl_applications",
            difficulty="advanced",
            title_ja="Self-Attention の QKV 分割",
            title_en="Split QKV for Self-Attention",
            description_ja="形状 [32, 128, 1536] のテンソルを Q, K, V の3つに分割してください。各々の形状は [32, 128, 512] です。",
            description_en="Split a tensor of shape [32, 128, 1536] into Q, K, V. Each has shape [32, 128, 512].",
            hint_ja="chunk(3, dim=-1) を使用します。",
            hint_en="Use chunk(3, dim=-1).",
            setup_code="qkv = torch.randn(32, 128, 1536)",
            solution_code="""Q, K, V = torch.chunk(qkv, 3, dim=-1)
# Return Q as result for verification
result = Q""",
            tags=["attention", "qkv", "nlp"],
        ),

        Problem(
            id="dl_010",
            category="dl_applications",
            difficulty="expert",
            title_ja="Rotary Position Embedding（簡易版）",
            title_en="Rotary Position Embedding (Simplified)",
            description_ja="形状 [32, 128, 512] のテンソルの各偶数次元とその次の奇数次元をペアとして、回転変換を適用します（簡易版：ペアを入れ替えるだけでOK）。",
            description_en="Apply rotation transformation to pairs of even and odd dimensions in a tensor of shape [32, 128, 512] (simplified: just swap pairs).",
            hint_ja="偶数と奇数の次元を分離し、入れ替えて cat します。",
            hint_en="Separate even and odd dimensions, swap, and cat.",
            setup_code="x = torch.randn(32, 128, 512)",
            solution_code="""x_even = x[..., 0::2]  # [32, 128, 256]
x_odd = x[..., 1::2]   # [32, 128, 256]
# Simple swap (real RoPE is more complex)
result = torch.zeros_like(x)
result[..., 0::2] = x_odd
result[..., 1::2] = x_even""",
            tags=["rope", "positional", "nlp", "advanced"],
        ),

        Problem(
            id="dl_011",
            category="dl_applications",
            difficulty="expert",
            title_ja="Grouped Convolution の入力準備",
            title_en="Prepare Input for Grouped Convolution",
            description_ja="形状 [32, 128, 56, 56] のテンソルを8グループに分割し、[32, 8, 16, 56, 56] に整形してください。",
            description_en="Split a tensor of shape [32, 128, 56, 56] into 8 groups and reshape to [32, 8, 16, 56, 56].",
            hint_ja="view を使ってグループ次元を挿入します。",
            hint_en="Use view to insert group dimension.",
            setup_code="x = torch.randn(32, 128, 56, 56)",
            solution_code="result = x.view(32, 8, 16, 56, 56)",
            tags=["grouped_conv", "cv", "reshape"],
        ),

        Problem(
            id="dl_012",
            category="dl_applications",
            difficulty="expert",
            title_ja="Label Smoothing の実装",
            title_en="Implement Label Smoothing",
            description_ja="形状 [32] のラベル（0~9）を、形状 [32, 10] の Label Smoothing された分布に変換してください（smoothing=0.1）。正解クラスは 0.9、他は 0.1/9。",
            description_en="Convert labels of shape [32] (0-9) to a label-smoothed distribution of shape [32, 10] (smoothing=0.1). Correct class gets 0.9, others get 0.1/9.",
            hint_ja="ones を作り、正解位置を scatter_ で調整します。",
            hint_en="Create ones and adjust correct positions with scatter_.",
            setup_code="""labels = torch.randint(0, 10, (32,))
smoothing = 0.1
num_classes = 10""",
            solution_code="""# Create smoothed distribution
result = torch.full((32, num_classes), smoothing / (num_classes - 1))
# Set correct class probability
result.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)""",
            tags=["label_smoothing", "ml", "scatter"],
        ),

        Problem(
            id="dl_013",
            category="dl_applications",
            difficulty="expert",
            title_ja="Feature Pyramid の構築",
            title_en="Build Feature Pyramid",
            description_ja="3つの異なる解像度の特徴マップ [32, 256, 56, 56]、[32, 512, 28, 28]、[32, 1024, 14, 14] を、すべて [32, 256, 56, 56] にアップサンプル/調整して結合してください。結果の形状は [32, 768, 56, 56] です。",
            description_en="Given 3 feature maps of different resolutions [32, 256, 56, 56], [32, 512, 28, 28], [32, 1024, 14, 14], upsample/adjust all to [32, 256, 56, 56] and concatenate. Result shape is [32, 768, 56, 56].",
            hint_ja="F.interpolate でアップサンプルし、必要に応じて channel を調整します（この問題では単純化のため concat のみ）。",
            hint_en="Use F.interpolate for upsampling (simplified: just assume channels are already adjusted).",
            setup_code="""feat1 = torch.randn(32, 256, 56, 56)
feat2 = torch.randn(32, 256, 28, 28)
feat3 = torch.randn(32, 256, 14, 14)""",
            solution_code="""# Upsample to same spatial size
feat2_up = F.interpolate(feat2, size=(56, 56), mode='bilinear', align_corners=False)
feat3_up = F.interpolate(feat3, size=(56, 56), mode='bilinear', align_corners=False)
# Concatenate
result = torch.cat([feat1, feat2_up, feat3_up], dim=1)""",
            tags=["fpn", "upsample", "cv", "interpolate"],
        ),

        Problem(
            id="dl_014",
            category="dl_applications",
            difficulty="expert",
            title_ja="Soft Attention Weight の正規化",
            title_en="Normalize Soft Attention Weights",
            description_ja="形状 [32, 8, 128, 128] の生のアテンションスコアに、形状 [32, 1, 1, 128] のマスク（0または1）を適用し、マスクされた位置を -inf にしてから softmax で正規化してください。",
            description_en="Apply a mask of shape [32, 1, 1, 128] (0 or 1) to raw attention scores of shape [32, 8, 128, 128], set masked positions to -inf, then normalize with softmax.",
            hint_ja="masked_fill で -inf を設定し、F.softmax します。",
            hint_en="Use masked_fill to set -inf, then F.softmax.",
            setup_code="""scores = torch.randn(32, 8, 128, 128)
mask = torch.randint(0, 2, (32, 1, 1, 128)).bool()""",
            solution_code="""# Invert mask: True means attend, False means mask
mask_inverted = ~mask
scores_masked = scores.masked_fill(mask_inverted, float('-inf'))
result = F.softmax(scores_masked, dim=-1)""",
            tags=["attention", "softmax", "masking", "nlp"],
        ),
    ]

    return problems
