# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python/ML project for transformer-based models.

## Build and Development Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies (once requirements.txt exists)
pip install -r requirements.txt

# Run tests (once pytest is configured)
pytest
pytest tests/test_specific.py -v  # Run single test file
pytest -k "test_name" -v  # Run specific test by name
```

## Architecture

*To be documented as the project develops.*
