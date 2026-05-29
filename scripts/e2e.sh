#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-check}"
case "$MODE" in
  check | update) ;;
  *)
    echo "usage: $0 [check|update]" >&2
    exit 2
    ;;
esac

if [[ -n "${CATENA_CMD:-}" ]]; then
  read -r -a CATENA <<< "$CATENA_CMD"
else
  CATENA=(cargo run -q -p catena-cli --)
fi
COMMON=(stdlib/core.hex stdlib/gpu.hex stdlib/gpu.proof.hex)
CASES="$ROOT/tests/e2e/cases.sh"
EXPECTED="$ROOT/tests/e2e/expected"
ACTUAL="$ROOT/target/e2e/actual"

TEST_NAMES=()
TEST_COMMANDS=()
TEST_INPUTS=()
TEST_EXPECTED=()
TEST_ARGS=()

test_case() {
  local name="$1"
  shift

  local command=""
  local input=""
  local expected=""
  local -a args=()

  while (($#)); do
    case "$1" in
      --command)
        command="$2"
        shift 2
        ;;
      --input)
        input="$2"
        shift 2
        ;;
      --expected)
        expected="$2"
        shift 2
        ;;
      --)
        shift
        args=("$@")
        break
        ;;
      *)
        echo "unknown test_case field for \`$name\`: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ -z "$command" || -z "$input" ]]; then
    echo "test_case \`$name\` must set --command and --input" >&2
    exit 2
  fi

  TEST_NAMES+=("$name")
  TEST_COMMANDS+=("$command")
  TEST_INPUTS+=("$input")
  TEST_EXPECTED+=("$expected")
  TEST_ARGS+=("${args[*]}")
}

# shellcheck source=../tests/e2e/cases.sh
source "$CASES"

run_catena() {
  echo "+ ${CATENA[*]} $*"
  "${CATENA[@]}" "$@"
}

run_output_case() {
  local output="$1"
  shift

  mkdir -p "$(dirname "$ACTUAL/$output")"
  echo "+ ${CATENA[*]} $* --output target/e2e/actual/$output"
  "${CATENA[@]}" "$@" --output "$ACTUAL/$output"
}

run_test_cases() {
  local i name command input expected raw_args
  local -a extra_args

  echo "Running e2e cases"
  for i in "${!TEST_NAMES[@]}"; do
    name="${TEST_NAMES[$i]}"
    command="${TEST_COMMANDS[$i]}"
    input="${TEST_INPUTS[$i]}"
    expected="${TEST_EXPECTED[$i]}"
    raw_args="${TEST_ARGS[$i]}"
    read -r -a extra_args <<< "$raw_args"

    echo "case: $name"
    if [[ -n "$expected" ]]; then
      run_output_case "$expected" "$command" "${COMMON[@]}" "$input" "${extra_args[@]}"
    else
      run_catena "$command" "${COMMON[@]}" "$input" "${extra_args[@]}"
    fi
  done
}

rm -rf "$ACTUAL"
mkdir -p "$ACTUAL"

run_test_cases

if [[ "$MODE" == "update" ]]; then
  rm -rf "$EXPECTED"
  mkdir -p "$(dirname "$EXPECTED")"
  cp -R "$ACTUAL" "$EXPECTED"
  echo "Updated e2e expected outputs in tests/e2e/expected"
else
  if [[ ! -d "$EXPECTED" ]]; then
    echo "Missing e2e expected outputs. Run \`make e2e-update\` to create them." >&2
    exit 1
  fi

  if ! diff -ru "$EXPECTED" "$ACTUAL"; then
    echo
    echo "E2E expected outputs differ." >&2
    echo "Fix the compiler output or run \`make e2e-update\` and commit the expected output changes." >&2
    exit 1
  fi

  echo "E2E expected outputs match"
fi

echo "Examples passed"
