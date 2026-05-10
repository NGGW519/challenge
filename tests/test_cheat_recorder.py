"""CheatCodeRecorder import + 메타데이터 검증.

ROS env 밖이라 실제 동작은 검증 못 함. RECORDED_TOPICS 일관성과 stub fallback만 확인.
"""

import pytest

from aic_model_pkg import CheatCodeRecorder as M


def test_recorded_topics_nonempty():
    assert len(M.RECORDED_TOPICS) > 0


def test_recorded_topics_format():
    for entry in M.RECORDED_TOPICS:
        assert isinstance(entry, tuple) and len(entry) == 2
        name, ttype = entry
        assert name.startswith("/")
        assert "/msg/" in ttype  # ROS 2 메시지 풀 네임스페이스


def test_recorded_topics_required_set():
    """Phase 2 학습에 반드시 필요한 토픽들이 포함됐는지."""
    names = {n for n, _ in M.RECORDED_TOPICS}
    required = {
        "/joint_states",
        "/fts_broadcaster/wrench",
        "/left_camera/image",
        "/center_camera/image",
        "/right_camera/image",
        "/aic_controller/pose_commands",
        "/tf",
        "/tf_static",
    }
    missing = required - names
    assert not missing, f"missing required topics: {missing}"


def test_recorded_topics_includes_ground_truth():
    """학습 시 ground_truth(/scoring/tf)는 기록 필요. 평가에선 절대 사용 금지."""
    names = {n for n, _ in M.RECORDED_TOPICS}
    assert "/scoring/tf" in names


def test_class_subclasses_cheatcode():
    """Stub 또는 실 CheatCode를 상속해야 함."""
    assert issubclass(M.CheatCodeRecorder, M.CheatCode)


def test_class_has_record_count_attr():
    """attribute 정의 확인 — 통계 트래킹 인터페이스."""
    # __init__ 호출엔 parent_node가 필요해 mock 주입.
    class _StubNode:
        pass
    rec = M.CheatCodeRecorder(_StubNode())
    assert isinstance(rec._record_count, dict)
    assert rec._record_count == {}


def test_insert_cable_without_record_dir_falls_through(monkeypatch):
    """AIC_RECORD_BAG_DIR 미설정 시 super().insert_cable()이 호출되어야 한다.
    Stub CheatCode는 NotImplementedError를 던지므로, ROS env 밖에서는 그게 정확한 신호."""
    monkeypatch.delenv("AIC_RECORD_BAG_DIR", raising=False)

    class _StubNode:
        pass
    rec = M.CheatCodeRecorder(_StubNode())

    with pytest.raises(NotImplementedError):
        rec.insert_cable(task=None, get_observation=None,
                         move_robot=None, send_feedback=None)
