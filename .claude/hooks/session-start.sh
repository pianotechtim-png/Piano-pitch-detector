#!/bin/bash
# SessionStart hook: install Python and Node dependencies so both test suites
# can run in Claude Code on the web sessions. Idempotent and non-interactive.
set -euo pipefail

# Only run in the remote (web) environment; local setups manage their own venv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Best-effort pip upgrade; never fatal (a distro-managed pip can't self-uninstall).
python -m pip install --upgrade pip || echo "pip upgrade skipped"

# requirements-dev.txt pulls in the runtime deps plus pytest/pytest-cov.
pip install -r requirements-dev.txt

# Node deps for the frontend (jsdom) test suite.
npm ci

# Make `import app` work from anywhere in the session (tests rely on it).
echo 'export PYTHONPATH="${CLAUDE_PROJECT_DIR:-.}"' >> "$CLAUDE_ENV_FILE"
