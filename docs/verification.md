# V1 Verification

The prototype is verified in two layers:

1. Local safety checks validate path-safe patch handling, patch application, deterministic test execution, Python syntax, and temporary Git worktree isolation.
2. GitHub Actions runs the complete pytest suite on Python 3.11 with the real LangGraph dependency installed from PyPI.

The next quality gate is a 10-bug synthetic benchmark with at least 8 autonomous repairs before introducing an orchestrator.
