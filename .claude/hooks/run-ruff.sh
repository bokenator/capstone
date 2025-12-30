#!/bin/bash

file_path=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)

# Only run ruff on Python files
if [[ ! "$file_path" =~ \.py$ ]]; then
    exit 0
fi

if [ -f "$file_path" ]; then
    echo "Running ruff on: $file_path"
    cd "$CLAUDE_PROJECT_DIR"
    uv run ruff check "$file_path" 2>&1 || true
fi

exit 0
