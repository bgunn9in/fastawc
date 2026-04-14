#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cmake "$@"

BUILD_DIR=""

while [[ $# -gt 0 ]]; do
	if [[ "$1" == "-B" ]]; then
		BUILD_DIR="$2"
		break
	fi
	shift
done

if [[ -n "$BUILD_DIR" && -f "$BUILD_DIR/compile_commands.json" ]]; then
	echo "Updating compile_commands.json..."

	if [[ -f "$ROOT_DIR/compile_commands.json" ]]; then
		echo "Delete old compile_commands.json..."
		rm "$ROOT_DIR/compile_commands.json"
	fi

	cp "$BUILD_DIR/compile_commands.json" "$ROOT_DIR/compile_commands.json"
	echo "compile_commands.json updated."
fi
