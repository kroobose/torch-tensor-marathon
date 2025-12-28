# Contributing to PyTorch Tensor Marathon

Thank you for your interest in contributing! This project welcomes contributions from everyone.

## How to Contribute

### Adding New Problems

1. **Choose a Category**: Decide which category your problem belongs to (or propose a new category)
2. **Create Problem Definition**: Add your problem to the appropriate file in `src/torch_tensor_marathon/problems/`
3. **Follow the Problem Format**:
   ```python
   Problem(
       id="category_###",  # Use sequential numbering
       category="category_name",
       difficulty="beginner|intermediate|advanced|expert",
       title_ja="Japanese Title",
       title_en="English Title",
       description_ja="Japanese Description",
       description_en="English Description",
       setup_code="import torch\n...",
       solution_code="result = ...",
       hint_ja="Optional Japanese hint",
       hint_en="Optional English hint",
   )
   ```

4. **Test Your Problem**: Ensure it works correctly
   ```bash
   uv run tensor-marathon --category your_category
   ```

### Code Style Guidelines

- **Comments**: Write all code comments in English
- **Docstrings**: Use English for all docstrings
- **UI Strings**: Add both Japanese and English translations to `i18n.py`
- **Formatting**: Run `ruff format` before committing
- **Type Hints**: Use type hints for all function signatures

### Testing Changes

```bash
# Run CLI to test
uv run tensor-marathon --lang en
uv run tensor-marathon --lang ja

# Run web server to test
uv run uvicorn web.app:app --reload
```

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test thoroughly (both CLI and web interface)
5. Commit with clear messages: `git commit -m "Add: description of changes"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request with a clear description

### Commit Message Guidelines

- Use present tense: "Add feature" not "Added feature"
- Use imperative mood: "Move cursor to..." not "Moves cursor to..."
- Limit first line to 72 characters
- Reference issues and PRs when relevant

### Areas for Contribution

- 🔢 **New Problems**: Add more tensor manipulation challenges
- 🌐 **Translations**: Improve or add new language support
- 🎨 **UI/UX**: Enhance the web or CLI interface
- 📚 **Documentation**: Improve guides, add examples, fix typos
- 🐛 **Bug Fixes**: Report and fix bugs
- ✨ **Features**: Propose and implement new features

### Questions or Issues?

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Be respectful and constructive in all interactions

## Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd torch_tensor_marathon

# Install dependencies
uv sync

# Run in development mode
uv pip install -e .

# Test CLI
uv run tensor-marathon

# Test web app
uv run uvicorn web.app:app --reload
```

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different perspectives and experiences

Thank you for contributing to PyTorch Tensor Marathon! 🎉
