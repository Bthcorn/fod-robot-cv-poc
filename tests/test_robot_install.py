"""The robot installs the wheel with numpy + cv2 and nothing else -- apt's, on the
Pi's system python3.11. Every test here runs the real modules in a subprocess with
the research stack blocked, which is what that install looks like from Python's
side: `sys.modules[name] = None` makes `import name` and `from name.x import y`
raise, lazy imports included.

A subprocess, not monkeypatch: test_train_cli.py has already imported
fodcv.cli.train into this process, and a reload resolves `from fodcv.research
import training` from the module cache -- a green test for a broken install.
"""

import subprocess
import sys

import pytest

BLOCK = ("import sys; "
         "sys.modules.update({m: None for m in ('ultralytics', 'torch', 'yaml', 'onnx')}); ")

ROBOT_SIDE = ("fodcv.paths", "fodcv.runtime.policy", "fodcv.runtime.vision",
              "fodcv.cli.camera_hailo", "fodcv.cli.robot_stub")


def base_install(code):
    return subprocess.run([sys.executable, "-c", BLOCK + code], capture_output=True, text=True)


def test_the_robot_side_imports_with_numpy_and_cv2_alone():
    """INTEGRATION.md 1's promise. A stray `import yaml` in vision.py passes every
    other test in this suite and dies on the Pi."""
    result = base_install("import " + ", ".join(ROBOT_SIDE))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cli", ["smoke_test", "train", "eval", "export", "bench_pi"])
def test_a_research_command_exits_with_an_install_hint(cli):
    """The other seven commands land on the robot's PATH too. A traceback is not
    an answer; one line saying what to install is."""
    result = base_install(f"import fodcv.cli.{cli}")
    assert result.returncode == 1, result.stderr
    assert "pip install 'fod-vision[research" in result.stderr
