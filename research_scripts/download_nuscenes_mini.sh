#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/data/nuscenes"
DOWNLOAD_DIR="${ROOT}/downloads"
ARCHIVE="${DOWNLOAD_DIR}/v1.0-mini.tgz"
URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

mkdir -p "${DOWNLOAD_DIR}"

echo "[Download]"
echo "url: ${URL}"
echo "archive: ${ARCHIVE}"
curl -L -C - --fail --retry 5 --retry-delay 5 "${URL}" -o "${ARCHIVE}"

echo
echo "[Archive]"
ls -lh "${ARCHIVE}"

echo
echo "[Extract]"
tar -xzf "${ARCHIVE}" -C "${ROOT}" --no-same-owner --skip-old-files

echo
echo "[Result]"
find "${ROOT}" -maxdepth 2 -type d | sort | sed -n '1,80p'
