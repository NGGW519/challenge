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

# 2026-05-12 Q3: lifecycle ACTIVATE 가 막혔던 zenoh / RMW / passwd 환경변수를
# 제출 컨테이너가 빠뜨리지 않도록 자동 검증.
section "submit Dockerfile zenoh / lifecycle env"
if [[ -f "$SUBMIT_DF" ]]; then
  # RMW_IMPLEMENTATION 은 반드시 rmw_zenoh_cpp 여야 함 (토킷 평가가 zenoh 통신).
  if grep -E "^ENV[[:space:]]+RMW_IMPLEMENTATION[[:space:]]*=?[[:space:]]*rmw_zenoh_cpp" "$SUBMIT_DF" > /dev/null; then
    printf "  RMW_IMPLEMENTATION=rmw_zenoh_cpp ✅\n"
  else
    printf "  ❌ RMW_IMPLEMENTATION 누락 또는 rmw_zenoh_cpp 가 아님\n"
    FAIL=1
  fi

  # ZENOH_ROUTER_CHECK_ATTEMPTS — 평가 컨테이너 router 늦게 떠도 무한 재시도.
  if grep -E "^ENV[[:space:]]+ZENOH_ROUTER_CHECK_ATTEMPTS" "$SUBMIT_DF" > /dev/null; then
    printf "  ZENOH_ROUTER_CHECK_ATTEMPTS set ✅\n"
  else
    printf "  ⚠️  ZENOH_ROUTER_CHECK_ATTEMPTS 미설정 — router 늦게 뜨면 model 죽을 수 있음\n"
  fi

  # AIC_MODEL_PASSWD 가 CHANGE_IN_PROD 그대로면 ACL 활성 환경에서 인증 실패 위험.
  # (eval-override.yaml 에 빈 값 또는 토킷이 정한 값으로 주입돼야 함.)
  if grep -E '^ENV[[:space:]]+AIC_MODEL_PASSWD[[:space:]]*=?[[:space:]]*"?CHANGE_IN_PROD"?[[:space:]]*$' "$SUBMIT_DF" > /dev/null; then
    if [[ "$STRICT" == "1" ]]; then
      printf "  ❌ AIC_MODEL_PASSWD=CHANGE_IN_PROD — 제출 전 실제 값 또는 빈 값(ACL off)으로 교체\n"
      FAIL=1
    else
      printf "  ⚠️  AIC_MODEL_PASSWD=CHANGE_IN_PROD (제출 직전 STRICT=1 에선 fail)\n"
    fi
  else
    printf "  AIC_MODEL_PASSWD plausible ✅\n"
  fi
fi

# eval-override.yaml 에도 같은 검증 — 컴포즈 경로로 제출 image 검증할 때 필요.
EVAL_OV="${ROOT}/docker/eval-override.yaml"
if [[ -f "$EVAL_OV" ]]; then
  if grep -q "AIC_ROUTER_ADDR" "$EVAL_OV"; then
    printf "  eval-override AIC_ROUTER_ADDR set ✅\n"
  else
    printf "  ⚠️  eval-override.yaml 에 AIC_ROUTER_ADDR 누락\n"
  fi
  if grep -q "RMW_IMPLEMENTATION:[[:space:]]*rmw_zenoh_cpp" "$EVAL_OV"; then
    printf "  eval-override RMW_IMPLEMENTATION=rmw_zenoh_cpp ✅\n"
  else
    printf "  ⚠️  eval-override.yaml 에 RMW_IMPLEMENTATION 누락\n"
  fi
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
