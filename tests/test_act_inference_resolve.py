"""ACTPlus 패턴: AIC_POLICY_LOCAL_DIR / arg / AIC_POLICY_HF_REPO 우선순위.

huggingface_hub 모듈은 vast.ai 인스턴스에선 install되지만 로컬엔 없을 수 있다.
없을 때를 위해 sys.modules에 mock 주입해 테스트가 어디서나 돌게 한다.
"""

import sys
import types
from unittest.mock import MagicMock

# huggingface_hub이 로컬에 없으면 mock으로 주입 (함수 내부 import 처리).
if "huggingface_hub" not in sys.modules:
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = MagicMock(side_effect=Exception("hf_hub not installed"))
    sys.modules["huggingface_hub"] = fake_hf

from aic_model_pkg.act_inference import resolve_ckpt_path


def test_returns_none_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.delenv("AIC_POLICY_LOCAL_DIR", raising=False)
    monkeypatch.delenv("AIC_POLICY_HF_REPO", raising=False)
    # HF default snapshot_download 도 실패하도록
    sys.modules["huggingface_hub"].snapshot_download = MagicMock(
        side_effect=Exception("offline"))
    assert resolve_ckpt_path(None) is None


def test_local_dir_takes_priority(monkeypatch, tmp_path):
    """AIC_POLICY_LOCAL_DIR가 최고 우선순위."""
    local = tmp_path / "baked"
    local.mkdir()
    (local / "final.pt").write_bytes(b"x")
    monkeypatch.setenv("AIC_POLICY_LOCAL_DIR", str(local))

    fake_arg = tmp_path / "arg.pt"
    fake_arg.write_bytes(b"y")

    out = resolve_ckpt_path(fake_arg)
    assert out == local  # LOCAL_DIR > arg


def test_arg_used_when_local_dir_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("AIC_POLICY_LOCAL_DIR", raising=False)
    arg_file = tmp_path / "arg.pt"
    arg_file.write_bytes(b"x")
    out = resolve_ckpt_path(arg_file)
    assert out == arg_file


def test_local_dir_set_but_path_missing_falls_through(monkeypatch, tmp_path):
    """LOCAL_DIR가 가리키는 경로가 없으면 다음 우선순위로."""
    monkeypatch.setenv("AIC_POLICY_LOCAL_DIR", str(tmp_path / "does_not_exist"))
    arg = tmp_path / "arg.pt"; arg.write_bytes(b"x")
    out = resolve_ckpt_path(arg)
    assert out == arg


def test_hf_repo_used_as_last_resort(monkeypatch, tmp_path):
    monkeypatch.delenv("AIC_POLICY_LOCAL_DIR", raising=False)
    monkeypatch.setenv("AIC_POLICY_HF_REPO", "test/repo")
    expected = tmp_path / "hf_snapshot"
    expected.mkdir()
    mock = MagicMock(return_value=str(expected))
    sys.modules["huggingface_hub"].snapshot_download = mock
    out = resolve_ckpt_path(None)
    assert out == expected
    mock.assert_called_once()
    assert mock.call_args.kwargs["repo_id"] == "test/repo"


def test_hf_failure_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("AIC_POLICY_LOCAL_DIR", raising=False)
    monkeypatch.setenv("AIC_POLICY_HF_REPO", "test/repo")
    sys.modules["huggingface_hub"].snapshot_download = MagicMock(
        side_effect=ConnectionError("no net"))
    out = resolve_ckpt_path(None)
    assert out is None
