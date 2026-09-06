#!/usr/bin/env bash
# Cut a release: one tag carrying the wheel, the model bundle and their checksums.
#
#   NOTES=artifacts/arg-bolts-4-n-640/SHIPPING_640_conf0001_2026-09-06/README.txt \
#     bash scripts/release.sh 0.3.0
#
# Run on the Mac, from a clean main, with `gh auth login` done once. NOTES is an
# optional file for the release body -- the SHIPPING_* README.txt beside the .hef
# already holds the mAP and latency numbers, so point at it rather than retyping.
#
# The default bundle is whatever `fodcv.paths.DEPLOY_HEF` names, so tar and code
# cannot drift; the alternates are the BUNDLES list below. On the default: a future compile that rewrites the canonical bench_int8_hailo_model/
# directory is not shipped until DEPLOY_HEF says so. Before anything is tagged the
# extracted bundle is loaded through the built wheel in a clean Python 3.11 -- the
# Pi's interpreter -- with only the wheel's own dependencies installed. That is the
# one check that fails if the layout, the metadata or the base deps are wrong.
#
# Nothing leaves this machine until the last three lines. A failure before them
# leaves only the version bump to revert.
set -euo pipefail
cd "$(dirname "$0")/.."

V=${1:?usage: release.sh <version>, e.g. 0.3.0}
git diff --quiet && git diff --cached --quiet || { echo "working tree is dirty -- commit first"; exit 1; }
BRANCH=$(git branch --show-current)
[ "$BRANCH" = main ] || [ "${RELEASE_BRANCH_OK:-}" = 1 ] || {
  echo "on $BRANCH, not main -- merge first, or RELEASE_BRANCH_OK=1 to tag here anyway"; exit 1; }
# A PR merged on GitHub puts commits on origin that this checkout lacks, and the
# push below is then rejected. Find out now, before the version bump.
git fetch -q origin
git merge-base --is-ancestor "origin/$BRANCH" HEAD || {
  echo "origin/$BRANCH has commits you do not -- git pull first"; exit 1; }

uv version "$V"
uv run pytest -q

rm -rf dist
uv build --wheel
WHL=dist/fod_vision-$V-py3-none-any.whl
[ -f "$WHL" ] || { echo "expected $WHL, got:"; ls dist; exit 1; }

RUN=$(uv run python -c 'from fodcv.paths import DEPLOY_RUN; print(DEPLOY_RUN)')
HEF=$(uv run python -c 'from fodcv.paths import DEPLOY_HEF; print(DEPLOY_HEF)')
# Every bundle on the release. The deploy build first, named by its run id and
# taken from DEPLOY_HEF so it cannot drift; then the alternates RESULT.md lists
# under "if the shipping build does not suit". name=path/to/best.hef. Each tar
# holds run.json two levels up and the .hef's own directory -- the layout Vision
# reads -- so the 480 build unpacks beside the default in the same run directory.
BUNDLES=(
  "$RUN=$HEF"
  "arg-bolts-4-n-640-at480=artifacts/arg-bolts-4-n-640/bench_int8_hailo_model_480/best.hef"
  "arg-bolts-4-s-640-a16=artifacts/arg-bolts-4-s-640/bench_int8_hailo_model/best.hef"
)
ASSETS=("$WHL")
for spec in "${BUNDLES[@]}"; do
  name=${spec%%=*}; hef=${spec#*=}; bundle=dist/$name.tar.gz
  # COPYFILE_DISABLE: no ._* AppleDouble entries for the Pi to trip over.
  COPYFILE_DISABLE=1 tar czf "$bundle" "$(dirname "$(dirname "$hef")")/run.json" "$(dirname "$hef")"
  # The gate. Clean 3.11, no project: just the wheel and what it declares, loading
  # the *extracted* bundle. Fails if the layout, the metadata or the base deps are wrong.
  TMP=$(mktemp -d)
  tar xzf "$bundle" -C "$TMP"
  (cd "$TMP" && uv run --isolated --no-project --python 3.11 --with "$OLDPWD/$WHL" -- python - "$hef" <<'PY'
import sys
from fodcv.paths import DEPLOY_HEF
from fodcv.runtime.vision import class_names, hef_imgsz
import fodcv.cli.camera_hailo, fodcv.cli.robot_stub  # the two Pi commands import clean
hef = sys.argv[1]
print(f"{hef}: {class_names(hef)} at {hef_imgsz(hef)}", "= DEPLOY_HEF" if hef == str(DEPLOY_HEF) else "")
PY
  )
  rm -rf "$TMP"
  ASSETS+=("$bundle")
done

(cd dist && shasum -a 256 *.whl *.tar.gz > SHA256SUMS && cat SHA256SUMS)

git commit -am "release: v$V"
git tag -a "v$V" -m "v$V"
git push origin HEAD   # the branch alone first: a rejected push must abort before the tag is public
git push origin "v$V"
gh release create "v$V" "${ASSETS[@]}" dist/SHA256SUMS --title "v$V" \
  ${NOTES:+--notes-file "$NOTES"} ${NOTES:---generate-notes}
