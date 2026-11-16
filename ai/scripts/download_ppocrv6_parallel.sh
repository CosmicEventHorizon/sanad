#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams"
readonly EXPECTED_SIZE=232651336
readonly CONNECTIONS=24
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly MODEL_PATH="${PROJECT_DIR}/ai/pretrained/PP-OCRv6_medium_rec_pretrained.pdparams"
readonly PARTS_DIR="${PROJECT_DIR}/ai/pretrained/.ppocrv6_parts"
readonly PREFIX_PATH="${PARTS_DIR}/prefix"
readonly ASSEMBLED_PATH="${PARTS_DIR}/assembled.pdparams"

mkdir -p "${PARTS_DIR}"

if [[ -f "${MODEL_PATH}" ]]; then
  cp "${MODEL_PATH}" "${PREFIX_PATH}"
else
  : > "${PREFIX_PATH}"
fi

readonly PREFIX_SIZE="$(stat -c '%s' "${PREFIX_PATH}")"
if (( PREFIX_SIZE > EXPECTED_SIZE )); then
  echo "Existing file is larger than the expected checkpoint." >&2
  exit 1
fi

if (( PREFIX_SIZE < EXPECTED_SIZE )); then
  readonly REMAINING=$((EXPECTED_SIZE - PREFIX_SIZE))
  readonly CHUNK_SIZE=$(((REMAINING + CONNECTIONS - 1) / CONNECTIONS))
  pids=()

  for ((index = 0; index < CONNECTIONS; index++)); do
    start=$((PREFIX_SIZE + index * CHUNK_SIZE))
    (( start >= EXPECTED_SIZE )) && break
    end=$((start + CHUNK_SIZE - 1))
    (( end >= EXPECTED_SIZE )) && end=$((EXPECTED_SIZE - 1))
    part_path="${PARTS_DIR}/part_$(printf '%02d' "${index}")"
    expected_part_size=$((end - start + 1))

    if [[ -f "${part_path}" ]] && (( $(stat -c '%s' "${part_path}") == expected_part_size )); then
      continue
    fi

    curl -4 -L --fail --retry 100 --retry-all-errors --retry-delay 2 \
      --range "${start}-${end}" \
      --output "${part_path}" "${MODEL_URL}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi

cp "${PREFIX_PATH}" "${ASSEMBLED_PATH}"
for part_path in "${PARTS_DIR}"/part_*; do
  [[ -e "${part_path}" ]] || continue
  dd if="${part_path}" of="${ASSEMBLED_PATH}" conv=notrunc oflag=append status=none
done

actual_size="$(stat -c '%s' "${ASSEMBLED_PATH}")"
if (( actual_size != EXPECTED_SIZE )); then
  echo "Assembled size ${actual_size} does not match ${EXPECTED_SIZE}." >&2
  exit 1
fi

"${PROJECT_DIR}/.venv-paddle/bin/python" -c \
  "import paddle; state = paddle.load('${ASSEMBLED_PATH}'); print(f'Validated {len(state)} tensors')"

mv "${ASSEMBLED_PATH}" "${MODEL_PATH}"
rm -rf "${PARTS_DIR}"
echo "PP-OCRv6 Medium training checkpoint is ready: ${MODEL_PATH}"
