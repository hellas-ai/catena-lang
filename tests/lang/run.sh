#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-check}"
case "$MODE" in
  check | update) ;;
  *)
    echo "usage: $0 [check|update]" >&2
    exit 2
    ;;
esac

CASES="$ROOT/tests/lang/cases"
EXPECTED="$ROOT/tests/lang/expected"
ACTUAL="$ROOT/target/catena-lang-tests/lang/actual"
REPORTS="$ROOT/target/catena-lang-tests/lang/reports"

COMMON=(
  catena-lang/stdlib/cmc.hex
  catena-lang/stdlib/value.hex
  catena-lang/stdlib/buf.hex
  catena-lang/stdlib/index.hex
  catena-lang/stdlib/data.hex
  catena-lang/stdlib/fn.hex
  catena-lang/stdlib/product.hex
  catena-lang/stdlib/gpu.hex
)

rm -rf "$ACTUAL" "$REPORTS"
mkdir -p "$ACTUAL" "$REPORTS"

for case_file in "$CASES"/*.hex; do
  name="$(basename "$case_file" .hex)"
  out="$ACTUAL/$name"
  report="$REPORTS/$name"

  mkdir -p "$out" "$report"
  echo "case: $name"

  set +e
  cargo run -q -p catena-lang -- "${COMMON[@]}" "$case_file" --output-dir "$report" \
    >"$out/stdout.txt" 2>"$out/stderr.txt"
  status=$?
  set -e

  printf '%s\n' "$status" >"$out/status.txt"

  if [[ -d "$report/gpu" ]]; then
    mkdir -p "$out/generated"
    cp -R "$report/gpu/." "$out/generated/"
  fi
done

if [[ "$MODE" == "update" ]]; then
  rm -rf "$EXPECTED"
  mkdir -p "$(dirname "$EXPECTED")"
  cp -R "$ACTUAL" "$EXPECTED"
  echo "Updated catena-lang expected outputs in tests/lang/expected"
else
  if [[ ! -d "$EXPECTED" ]]; then
    echo "Missing expected outputs. Run \`tests/lang/run.sh update\`." >&2
    exit 1
  fi

  diff -ru "$EXPECTED" "$ACTUAL"
  echo "catena-lang regression outputs match"
fi
