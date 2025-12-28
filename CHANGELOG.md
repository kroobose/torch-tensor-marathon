# Changelog

All notable changes to PyTorch Tensor Marathon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-28

### Added
- **Web Application**: Modern web interface with FastAPI backend
  - Beautiful dark-themed UI with glassmorphism effects
  - Real-time code checking and feedback
  - Progress tracking with localStorage
  - Bilingual support (Japanese/English)
  - Responsive design for mobile devices
  - Instructions and rules on welcome screen
  - Improved error message formatting
  - Reset progress functionality
  - Home navigation via logo click

- **100 Curated Problems** across 8 categories:
  - 🔄 Reshape & Permute (15 problems)
  - 🎯 Indexing & Slicing (15 problems)
  - 📡 Broadcasting (12 problems)
  - 🎲 Gather & Scatter (10 problems)
  - ∑ Einstein Summation (12 problems)
  - 📚 Stacking & Splitting (10 problems)
  - ⚡ Advanced Operations (12 problems)
  - 🧠 Deep Learning Applications (14 problems)

- **Interactive CLI**:
  - Rich terminal UI with syntax highlighting
  - Category-based navigation
  - Problem browser with difficulty filters
  - Instant solution validation
  - Bilingual interface (Japanese/English)
  - Progress tracking
  - Hint system

- **Core Features**:
  - Automated correctness checking (shape and values)
  - `torch.allclose` validation with configurable tolerances
  - Comprehensive error messages
  - Solution code viewing
  - Setup code execution

### Documentation
- Bilingual README with installation and usage instructions
- Contributing guidelines
- MIT License
- Architecture documentation
- Web application documentation
- Code of Conduct

### Infrastructure
- Python 3.10+ support
- UV package manager integration
- FastAPI web server
- Modern frontend (Vanilla JS, Prism.js for syntax highlighting)
- Modular problem bank system

## [Unreleased]

### Planned
- Linear algebra operations category (PCA, SVD, matrix decomposition)
- User authentication for web app
- Code execution history
- Social sharing features
- Docker deployment configuration
- Advanced code editor (Monaco Editor integration)
- Real-time collaboration features

---

[1.0.0]: https://github.com/your-username/torch_tensor_marathon/releases/tag/v1.0.0
