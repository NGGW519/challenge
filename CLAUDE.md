# Claude Code 세션 가이드 (aic_work)

이 파일은 Claude Code가 새 세션을 시작할 때 자동으로 읽힌다.
**먼저 `reports/`의 가장 최신 파일을 읽고**, 그 다음 사용자에게 무엇을 할지 묻는다.

---

## 프로젝트 한 줄

[AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) 출전 — UR5e + 3 wrist cam + F/T 센서로 케이블(SFP/SC) 삽입을 ROS 2 Kilted / Gazebo에서 수행. 목표 **280+/300**.

---

## 핵심 규칙 (매 세션 적용)

1. **`/home/nggw/challenge/aic/` 는 read-only 토킷.** 절대 수정 금지. 참조만.
2. **모든 작업은 `/home/nggw/challenge/aic_work/`에서.** GitHub 백업은 `https://github.com/NGGW519/challenge.git` (`main`).
3. **사용자는 한국어로 답함.** 답변도 한국어, 기술명/명령/경로는 영어.
4. **사용자는 1인 단독.** 전략에 "team"이 나오면 Claude 서브에이전트 페르소나(`strategy/01_agent_team.md`).
5. **vast.ai 비용 무제한** — 시간이 더 비싸다. idle 인스턴스만 정지.
6. **돈 들지 않는 로컬 작업을 먼저** — 인스턴스 spin-up이나 ECR push처럼 외부 의존이 큰 작업은 사용자 동의 후.
7. **커밋 메시지는 한국어 한 줄, Claude attribution 없음.** `Co-Authored-By` 트레일러 금지. push는 보통 사용자가 직접.

---

## 다음 세션을 시작할 때 체크리스트

### 1. 최신 상태 파악
```bash
ls -1 reports/             # 최신 파일이 가장 최근 진척
git log --oneline -10
```
**가장 먼저 `reports/<최근 날짜>.md`를 끝까지 읽는다.** 거기 §6 (다음 세션 후보)에 즉시 들어갈 수 있는 작업이 있다.

### 2. 검증 baseline 확인 (선택, 회귀 의심 시)
```bash
python3 -m pytest tests/ -q              # 20 PASS 기대
python3 -m ruff check aic_model_pkg/ scripts/ data/ training/ tests/
bash scripts/audit_pre_submit.sh         # PASS 기대 (non-strict)
```

### 3. 인프라 자원 상태 확인 (사용자에게 물어봄)
- vast.ai 인스턴스가 떠 있는가?
- R2 / HF Hub 자격증명이 환경에 들어가 있는가?
- 아니라면 무료 로컬 작업 (HybridPolicy 코드 채우기, CI 추가, strategy 결정 로그 등)을 우선 제안.

---

## 디렉토리 1줄 가이드

```
strategy/   ─ 17개 전략 문서. 작업 시작 전 해당 Phase 문서를 우선 참조.
reports/    ─ 일자별 진척 보고서. 새 작업 세션 종료 시 새 파일 추가.
docker/     ─ dev / submit Dockerfile + compose override.
scripts/    ─ vast.ai 부팅, 평가, 학습 wrapper, 집계, 시각화, 사전 audit.
data/       ─ randomize_scene (동작 검증) + bag_to_lerobot / auto_label_ports (ROS env 필요).
training/   ─ train_act.py, yolov8_port.yaml.
aic_model_pkg/  ─ ROS 2 ament_python 패키지. 제출 컨테이너의 진입점.
tests/      ─ pytest 20개 (geometry / safety / randomize_scene). 변경 시 깨지지 않게 유지.
```

---

## 자주 쓰는 명령

```bash
# 시나리오 무작위 trial 생성
python3 data/randomize_scene.py --scenario sfp1 --seed 42

# 베이스라인 평가 (vast.ai 인스턴스 안에서 실행)
bash scripts/run_baseline.sh CheatCode 5

# 우리 정책 100 rollout
AIC_SUBMIT_TAG=v1 bash scripts/eval_100.sh hybrid_v1 100 4

# 학습 (자동 재개)
bash scripts/train_with_resume.sh act

# 제출 직전 점검
STRICT=1 bash scripts/audit_pre_submit.sh
```

---

## 주의 — 챌린지 규칙 위반 패턴

`audit_pre_submit.sh`가 자동으로 잡지만 작성 단계에서도 의식할 것:

- `/scoring/*`, `/gazebo/*`, `/clock`, `/model*` 의 파라미터 수정 금지
- `gz topic pub`로 직접 텔레포트 금지
- `/scoring/tf` 구독은 **학습 코드에만**, 제출 컨테이너에는 절대 포함 금지
- 외부 네트워크 호출 금지
- aic_model lifecycle: 60초 안에 configure/active 도달, `unconfigured`/`configured` 상태에서 publish 금지

위반 시 Tier 1 = 0 → 전체 점수 0.

---

## 사용자가 별도 준비해야 하는 자원 (대기 중)

`reports/2026-05-10.md` §5 참조. 요약:
1. vast.ai 계정 + 잔고
2. Cloudflare R2 bucket + access key
3. HuggingFace Hub token + private repo
4. AWS ECR 자격증명 (조직 측 이메일)
5. 제출 포털 자격증명

`#1+#2+#3` 이 들어오면 학습/데이터 작업 즉시 시작 가능.
