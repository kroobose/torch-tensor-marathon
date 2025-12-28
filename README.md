# 🏃 PyTorch Tensor Marathon

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive practice system designed to help you master PyTorch tensor shape manipulations through **100 curated problems** across 8 categories.

Whether you are a beginner looking to understand `reshape` and `view`, or an expert implementing complex attention mechanisms with `einsum`, this repository has something for you.

## ✨ Features

- **100 Curated Problems**: Carefully designed progression from basic shape changes to complex Deep Learning patterns.
- **8 Categories**: Covers everything from basic indexing to advanced Einstein summation and DL applications (ViT patches, Attention masks, RoPE, etc.).
- **Dual Interface**:
  - **Interactive CLI**: A beautiful, terminal-based interface with syntax highlighting.
  - **Modern Web App**: A responsive, glassmorphism-styled web interface with **Light/Dark themes** and progress tracking.
- **Auto-Correction**: Validates your solution by checking both **tensor shapes** and **values** using `torch.allclose`.
- **AI-Powered Assistance**: Integrated with **Google Gemini** to provide:
  - **AI Explanations**: Understand *why* a solution works.
  - **AI Hints**: Get nudges in the right direction without revealing the full answer.

## 📦 Installation

This project uses `uv` for modern Python package management.

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/torch_tensor_marathon.git
cd torch_tensor_marathon

# 2. Install dependencies
uv sync

# Alternatively, using pip:
# pip install -e .
```

## 🚀 Usage

You can run the application in two modes: **CLI** or **Web**.

### Option A: Web Application (Recommended)

The web interface offers the best experience with syntax highlighting, visual feedback, and AI integration.

```bash
# Start the web server
uv run uvicorn web.app:app --reload
```

Then open your browser to: **http://localhost:8000**

**Web Features:**
- **Theme Toggle**: Switch between 🌙 Dark and ☀️ Light modes.
- **Progress Tracking**: Your solved problems are saved automatically in the browser.
- **Bilingual UI**: One-click switch between English and Japanese interfaces.

### Option B: Interactive CLI

Perfect for quick practice in the terminal.

```bash
# Run the text-based interface
uv run tensor-marathon
```

**CLI Options:**
```bash
# Jump directly to a category
uv run tensor-marathon --category einsum

# Force specific language (default is English)
uv run tensor-marathon --lang ja
```

## 🧠 AI Features (Gemini)

To enable AI explanations and hints, you need a Google Gemini API key.

1.  **Get an API Key**: Visit [Google AI Studio](https://aistudio.google.com/).
2.  **Set the Environment Variable**:

    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```
3.  **Run the App**: The AI buttons (🤖 Explanation, 💡 Hint) will automatically appear in the web interface.

*Note: You can also set the API key directly within the Web UI settings.*

## 📚 Problem Categories

| Category | Count | Description |
|----------|-------|-------------|
| **Reshape & Permute** | 15 | `view`, `reshape`, `permute`, `transpose`, contiguous memory |
| **Indexing & Slicing** | 15 | Fancy indexing, boolean masking, `torch.where`, ellipsis |
| **Broadcasting** | 12 | `unsqueeze`, `expand`, broadcasting rules, normalization |
| **Gather & Scatter** | 10 | `torch.gather`, `torch.scatter_add_`, one-hot encoding |
| **Einstein Summation** | 12 | `torch.einsum` for matrix multiplication, attention, contractions |
| **Stacking & Splitting** | 10 | `torch.cat`, `torch.stack`, `torch.chunk`, `torch.split` |
| **Advanced Operations** | 12 | Masking, sorting, `topk`, padding, triangular matrices |
| **DL Applications** | 14 | Real-world patterns: ViT patches, Multi-head Attention, Mixup, Focal Loss, RoPE |

**Total:** 100 Problems

## 🤝 Contributing

Contributions are welcome! If you have a new problem idea or want to improve an existing one, please feel free to open a Pull Request.

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/amazing-problem`)
3.  Commit your changes (`git commit -m 'Add new problem about LoRA'`)
4.  Push to the branch (`git push origin feature/amazing-problem`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
