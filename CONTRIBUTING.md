# Contributing to Customer Churn Prediction Model

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### 1. Fork the Repository

Click the "Fork" button on the top right of the repository page to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/hithesh-20/Predictive-Customer-Churn-Model.git
cd Predictive-Customer-Churn-Model
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 5. Make Your Changes

- Write clean, well-documented code
- Add comments for complex logic
- Update tests if applicable
- Update documentation as needed

### 6. Test Your Changes

```bash
python src/train.py
```

### 7. Commit Your Changes

```bash
git add .
git commit -m "Add: description of your changes"
```

### 8. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 9. Create a Pull Request

Go to the original repository on GitHub and click "New Pull Request".

## Code Style Guidelines

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

## Reporting Issues

If you find a bug or have a feature request, please create an issue on GitHub with:
1. A clear description of the issue
2. Steps to reproduce (if applicable)
3. Expected vs actual behavior
4. Environment details (Python version, OS, etc.)

## Questions?

Feel free to open an issue for any questions or clarifications.