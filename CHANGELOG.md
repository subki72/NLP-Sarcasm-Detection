# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-18

### Added
- Fine-tuned DistilBERT sarcasm detection model inference API with FastAPI.
- Robust rate limiting using SlowAPI and Prometheus observability metrics.
- High-coverage automated pytest suite and container smoke tests in CI.
- Multi-aspect CI validation including Ruff linting, test suite, and Trivy security scanning.
- Pinned deterministic lockfiles (`requirements-lock.txt`, `uv.lock`) and 1-command Makefile.
- Standard governance documentation (`SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`).
- Structured leveled logging across all inference pipelines.
