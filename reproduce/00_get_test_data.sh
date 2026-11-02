#!/usr/bin/env bash
# Stage 00 (CPU): download and verify the released test package (20 clips, 1.3 GB) from macvi.org -> 1_data/raw/lookout-vo-test-v0.1
. "$(dirname "$0")/common.sh"
mkdir -p 1_data/raw && cd 1_data/raw; f=lookout-vo-test-v0.1.tar
[ -f lookout-vo-test-v0.1/release_manifest.json ] && { echo "test package already extracted ($(ls lookout-vo-test-v0.1/videos 2>/dev/null | wc -l) videos)"; exit 0; }
[ -f "$f" ] || curl -fL -C - -O "https://macvi.org/downloads/lookout/v0.1/$f"
curl -fsSL -O "https://macvi.org/downloads/lookout/v0.1/$f.sha256" && sha256sum -c "$f.sha256" 2>/dev/null || shasum -a 256 -c "$f.sha256"
[ -d lookout-vo-test-v0.1 ] || tar -xf "$f"; ls lookout-vo-test-v0.1/videos | wc -l
