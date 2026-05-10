#!/usr/bin/env bash
# 제출 직전 점검 — 챌린지 규칙 위반 패턴 검색.
# 통과해야 ECR push 진행.
#
# 사용:
#   bash scripts/audit_pre_submit.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/aic_model_pkg/"
FAIL=0

STRICT="${STRICT:-0}"  # 1 = 제출 직전 모드 (weights 파일 누락도 fail)
TRAINING_ONLY_MARKER="AIC-TRAINING-ONLY"

section() { printf "\n== %s ==\n" "$1"; }

# .py 파일만 검사. AIC-TRAINING-ONLY 마커가 있는 파일은 학습 전용이라
# 제출 이미지에 포함되지 않으므로 검사에서 제외 (.dockerignore가 빌드 시 차단).
check_clean() {
  local label="$1"; shift
  local raw filtered
  raw=$(grep -rEn --include='*.py' "$@" "$SRC" 2>/dev/null || true)
  filtered=""
  if [[ -n "$raw" ]]; then
    while IFS= read -r line; do
      local file="${line%%:*}"
      [[ -z "$file" ]] && continue
      if grep -q "$TRAINING_ONLY_MARKER" "$file" 2>/dev/null; then
        continue   # 학습 전용 — 제외
      fi
      filtered+="${line}"$'\n'
    done <<< "$raw"
  fi
  if [[ -n "$filtered" ]]; then
    printf "  ❌ FOUND:\n%s" "$filtered"
    FAIL=1
  else
    printf "  ✅ %s — clean\n" "$label"
  fi
}

section "forbidden ROS topic/param manipulations"
check_clean "no /scoring/* /gazebo/* /clock /model* writes" \
  "set_parameters?\(.*['\"](\/scoring|\/gazebo|\/gz_server|\/clock|\/model)"
check_clean "no force-publish to scoring topics" \
  "create_publisher.*['\"]\/scoring"

section "forbidden teleport / direct sim manipulation"
check_clean "no gz topic pub teleport" "gz topic pub|gz model.*--pose|gz service.*set_pose"

section "ground-truth (TF /scoring/tf) only in training"
check_clean "no /scoring/tf subscribe in shipped code" \
  "['\"]\/scoring\/tf['\"]"

section "no external network calls"
check_clean "no requests/urlopen/socket" \
  "(requests\.get|requests\.post|urlopen|socket\.connect|http://|https://)"

section "no lifecycle violations (publish before active)"
# 약식: on_configure 안에서 직접 publish 호출 패턴
check_clean "no publish in on_configure" \
  "def on_configure[^}]*self\.[a-zA-Z_]+_pub\.publish"

section "weight files present"
WEIGHTS_DIR="${SRC}aic_model_pkg/weights"
if [[ -d "$WEIGHTS_DIR" ]]; then
  shopt -s nullglob
  files=("$WEIGHTS_DIR"/*.pt "$WEIGHTS_DIR"/*.pth "$WEIGHTS_DIR"/*.bin "$WEIGHTS_DIR"/*.safetensors)
  shopt -u nullglob
  if [[ ${#files[@]} -gt 0 ]]; then
    printf "  ✅ weights:\n"
    for f in "${files[@]}"; do printf "     %s (%s)\n" "$f" "$(du -h "$f" | cut -f1)"; done
  else
    if [[ "$STRICT" == "1" ]]; then
      printf "  ❌ no weight files in %s — submit container will be useless\n" "$WEIGHTS_DIR"
      FAIL=1
    else
      printf "  ℹ️  no weight files in %s yet (run with STRICT=1 to fail before submission)\n" "$WEIGHTS_DIR"
    fi
  fi
else
  if [[ "$STRICT" == "1" ]]; then
    printf "  ❌ weights directory missing: %s\n" "$WEIGHTS_DIR"; FAIL=1
  else
    printf "  ℹ️  weights directory missing: %s (placeholder ok before training)\n" "$WEIGHTS_DIR"
  fi
fi

section "Dockerfile sanity"
SUBMIT_DF="${ROOT}/docker/submit.Dockerfile"
if [[ -f "$SUBMIT_DF" ]]; then
  printf "  submit.Dockerfile present ✅\n"
  if grep -q "AIC_MODEL_PASSWD" "$SUBMIT_DF"; then
    printf "  AIC_MODEL_PASSWD set ✅\n"
  else
    printf "  ⚠️  AIC_MODEL_PASSWD missing in Dockerfile\n"
  fi
else
  printf "  ❌ submit.Dockerfile missing\n"; FAIL=1
fi

if [[ $FAIL -eq 0 ]]; then
  echo
  echo "✅ pre-submit audit PASSED"
  exit 0
else
  echo
  echo "❌ pre-submit audit FAILED — fix above before push"
  exit 1
fi
