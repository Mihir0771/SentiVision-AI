# Contributing to SentiVision AI

First off, **thank you** for considering contributing to SentiVision AI! 🎉

Every contribution helps make this project better — whether it's a bug fix, a new feature, improved documentation, or even just a typo correction.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#-reporting-bugs)
  - [Suggesting Features](#-suggesting-features)
  - [Code Contributions](#-code-contributions)
- [Development Setup](#-development-setup)
- [Code Style](#-code-style)
- [Pull Request Process](#-pull-request-process)
- [Commit Message Conventions](#-commit-message-conventions)

---

## Code of Conduct

This project follows a standard open-source Code of Conduct. Please be respectful, inclusive, and constructive in all interactions.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Found a bug? Help us fix it by opening an issue.

**Before submitting**, please:

1. **Search existing issues** to see if it's already been reported.
2. **Check the [Troubleshooting](README.md#troubleshooting) section** in the README.
3. **Verify** the bug exists on the latest version of `main`.

**When submitting**, include:

- **Clear title** describing the bug
- **Steps to reproduce** — be as specific as possible
- **Expected behavior** vs. **actual behavior**
- **Environment details**:
  - OS and version
  - Python version (`python --version`)
  - Browser and version (for UI bugs)
  - MongoDB version
- **Screenshots / logs** if applicable
- **Error traceback** (full traceback, not just the final line)

Use this template:

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. Enter '...'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: Windows 11
- Python: 3.11.5
- Browser: Chrome 120
- MongoDB: 7.0
```

### 💡 Suggesting Features

We welcome feature ideas! Open an issue with the **feature request** label.

Please include:

- **Problem statement** — What problem does this solve?
- **Proposed solution** — How should it work?
- **Alternatives considered** — Other approaches you thought about
- **Additional context** — Mockups, examples, or references

### 💻 Code Contributions

Ready to write code? Here's how to get started:

1. **Fork** the repository
2. **Create a branch** from `main` for your work
3. **Write your code** following our [style guide](#-code-style)
4. **Write tests** for your changes
5. **Submit a pull request**

---

## 🔧 Development Setup

### Prerequisites

- Python 3.9 or higher
- MongoDB 6.0+ running locally or in Docker
- Git

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/sentivision-ai.git
cd sentivision-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies (including dev extras)
pip install -r requirements.txt
pip install -e ".[dev]"

# 4. Copy environment template
cp .env.example .env
# Edit .env with your local settings

# 5. Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# 6. Seed the database (optional)
python scripts/seed_db.py

# 7. Run the app
python run.py

# 8. Run tests
pytest --cov=app tests/
```

### Project Layout

```
├── app/                # Application package
│   ├── models/         # Data models
│   ├── routes/         # Route blueprints
│   ├── services/       # Business logic & AI services
│   ├── static/         # CSS, JS, images
│   └── templates/      # Jinja2 HTML templates
├── tests/              # Test suite
├── scripts/            # Utility scripts
└── run.py              # Application entry point
```

---

## 📝 Code Style

We follow strict Python coding standards to maintain a clean, readable codebase.

### General Rules

| Rule | Standard |
|------|----------|
| **Formatter** | [Black](https://github.com/psf/black) (line length 100) |
| **Import sorting** | [isort](https://pycqa.github.io/isort/) (Black-compatible profile) |
| **Linter** | [flake8](https://flake8.pycqa.org/) |
| **Type hints** | Required for all public functions |
| **Docstrings** | Google style, required for all public classes/functions |

### Python Style

```python
# ✅ Good
def analyze_sentiment(text: str, model: str = "ensemble") -> dict[str, Any]:
    """Analyze the sentiment of the given text.

    Args:
        text: The input text to analyze.
        model: The model to use. One of 'vader', 'textblob',
               'roberta', or 'ensemble'.

    Returns:
        A dictionary containing sentiment scores, label, and confidence.

    Raises:
        ValueError: If the text is empty or the model is invalid.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    ...


# ❌ Bad
def analyze(t, m="ensemble"):
    # analyze text
    if not t:
        return None
    ...
```

### Running Linters Locally

```bash
# Format code
black --line-length 100 app/ tests/

# Sort imports
isort --profile black app/ tests/

# Lint
flake8 app/ tests/ --max-line-length 100

# Type check (optional)
mypy app/ --ignore-missing-imports
```

---

## 🔀 Pull Request Process

### Before You Submit

1. ✅ Code follows our [style guide](#-code-style)
2. ✅ All existing tests pass (`pytest`)
3. ✅ New tests added for new functionality
4. ✅ Documentation updated if needed
5. ✅ No merge conflicts with `main`

### PR Guidelines

- **One feature / fix per PR** — Keep changes focused
- **Write a clear description** explaining *what* and *why*
- **Reference related issues** using `Closes #123` or `Fixes #456`
- **Keep commits clean** — Squash WIP commits before submitting
- **Add screenshots** for UI changes

### PR Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran.

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
- [ ] I have updated documentation accordingly
```

### Review Process

1. A maintainer will review your PR within **48 hours**
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Your contribution will appear in the next release 🚀

---

## 📌 Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, machine-readable history.

### Format

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `style` | Code style changes (formatting, semicolons, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `build` | Changes to build system or dependencies |
| `ci` | Changes to CI configuration |
| `chore` | Other changes that don't modify src or test files |

### Examples

```bash
feat(analysis): add keyword extraction with RAKE
fix(camera): resolve webcam permission error on Chrome
docs(readme): update installation instructions
test(api): add tests for batch analysis endpoint
refactor(services): extract ensemble scoring into separate module
```

---

## 🙏 Thank You

Your contributions make SentiVision AI better for everyone. We truly appreciate your time and effort!

If you have questions, feel free to open a discussion or reach out via issues.

Happy coding! 🚀
