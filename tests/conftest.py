import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _radiance_bin():
    rdir = os.environ.get("RADIANCE_DIR")
    if rdir and (Path(rdir) / "bin").exists():
        return Path(rdir) / "bin"
    exe = shutil.which("gendaylit")
    return Path(exe).parent if exe else None


@pytest.fixture(scope="session")
def radiance():
    """Runner for Radiance programs: radiance(["gendaylit", ...], stdin=None, cwd=None) -> stdout."""
    bin_dir = _radiance_bin()
    if bin_dir is None:
        pytest.skip("Radiance not found (set RADIANCE_DIR or put Radiance/bin on PATH)")
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    lib = bin_dir.parent / "lib"
    env["RAYPATH"] = os.pathsep.join(["."] + ([str(lib)] if lib.exists() else []))

    def run(cmd, stdin=None, cwd=None):
        exe = shutil.which(cmd[0], path=env["PATH"])
        p = subprocess.run([exe or cmd[0]] + [str(c) for c in cmd[1:]], input=stdin, cwd=cwd, env=env,
                           capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed: {p.stderr.decode(errors='replace')}")
        return p.stdout.decode()

    return run
