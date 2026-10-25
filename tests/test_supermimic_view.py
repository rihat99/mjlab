"""tools/view.py: the clip file and the browser it picks."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
  """The ``view`` and ``render_viser`` modules of tools/ (not a package)."""
  monkeypatch.syspath_prepend(str(TOOLS))
  return importlib.import_module("view"), importlib.import_module("render_viser")


def test_resolve_motion(tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  view, _ = tools
  recorded, seed, lafan = (tmp_path / name for name in ("recorded", "seed", "lafan"))
  for folder, clips in ((recorded, "a"), (seed, "ab"), (lafan, "c")):
    folder.mkdir()
    for clip in clips:
      (folder / f"{clip}.npz").touch()
  monkeypatch.setattr(view, "MOTIONS_SEED_DIR", seed)
  monkeypatch.setattr(view, "MOTIONS_DIR", lafan)
  run = tmp_path / "run"
  (run / "params").mkdir(parents=True)
  listing = tmp_path / "library_files.txt"
  listing.write_text(f"{recorded / 'a.npz'}\n")

  def record(motion_file: str) -> None:
    (run / "params" / "env.yaml").write_text(
      f"commands:\n  motion:\n    motion_file: {motion_file}\n"
    )

  record(str(listing))
  # The run's own motions dir first, then SEED, then LAFAN.
  assert view.resolve_motion("a", None, run) == recorded / "a.npz"
  assert view.resolve_motion("b", None, run) == seed / "b.npz"
  assert view.resolve_motion("c", None, run) == lafan / "c.npz"
  assert view.resolve_motion("a", seed, run) == seed / "a.npz"
  with pytest.raises(SystemExit, match="no c.npz"):
    view.resolve_motion("c", recorded, run)
  # A run made on another machine: its clip list is not here.
  record(str(tmp_path / "elsewhere" / "library_files.txt"))
  assert view.resolve_motion("a", None, run) == seed / "a.npz"
  # A library recorded as a tuple of npz paths.
  record(f"!!python/tuple\n    - {recorded / 'a.npz'}")
  assert view.resolve_motion("a", None, run) == recorded / "a.npz"


def test_chrome(tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  _, render_viser = tools
  for version in (99, 1243):
    shell = tmp_path / render_viser.PLAYWRIGHT_SHELL.replace("*", str(version))
    shell.parent.mkdir(parents=True)
    shell.write_text(
      "#!/bin/sh\necho 'error while loading shared libraries: libasound.so.2: "
      "cannot open shared object file' >&2\nexit 127\n"
    )
    shell.chmod(0o755)
  monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
  chrome = render_viser.find_chrome()
  assert "chromium_headless_shell-1243" in chrome  # the newest by number
  with pytest.raises(SystemExit, match="apt-get download libasound2t64"):
    render_viser.check_chrome(chrome)
