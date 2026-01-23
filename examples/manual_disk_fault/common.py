import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import IO, Optional

DM_DEV_NAME = "demo_disk"
MOUNT_POINT = "/mnt/demo_disk"
DM_DEV_PATH = f"/dev/mapper/{DM_DEV_NAME}"

def run_cmd(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    input: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    print(f"Running: {' '.join(shlex.quote(a) for a in args)}")
    return subprocess.run(
        args,
        text=True,
        check=check,
        capture_output=capture_output,
        input=input,
    )

def find_scsi_debug_device() -> str:
    def _is_scsi_debug(model: str, vendor: str) -> bool:
        m = (model or "").strip().lower()
        v = (vendor or "").strip().lower()
        if "scsi_debug" in m:
            return True
        if v == "linux" and "scsi" in m and "debug" in m:
            return True
        return False

    try:
        proc = run_cmd(
            ["lsblk", "-J", "-o", "NAME,TYPE,VENDOR,MODEL"],
            check=True,
            capture_output=True,
        )
        data = json.loads(proc.stdout or "{}")
        for dev in data.get("blockdevices", []) or []:
            if (dev.get("type") or "").strip() != "disk":
                continue
            name = (dev.get("name") or "").strip()
            if not name:
                continue
            if _is_scsi_debug(dev.get("model") or "", dev.get("vendor") or ""):
                return f"/dev/{name}"
    except Exception:
        pass
    
    # Fallback to sysfs scan
    try:
        sys_block = Path("/sys/block")
        if sys_block.exists():
            for devdir in sorted(sys_block.iterdir()):
                name = devdir.name
                model_path = devdir / "device" / "model"
                vendor_path = devdir / "device" / "vendor"
                try:
                    model = model_path.read_text(encoding="utf-8", errors="ignore") if model_path.exists() else ""
                    vendor = vendor_path.read_text(encoding="utf-8", errors="ignore") if vendor_path.exists() else ""
                except OSError:
                    continue
                if _is_scsi_debug(model, vendor):
                    return f"/dev/{name}"
    except Exception:
        pass

    raise RuntimeError("Cannot find scsi_debug device. Is the module loaded?")

def get_device_sectors(device_path: str) -> str:
    proc = run_cmd(["blockdev", "--getsz", device_path], check=True, capture_output=True)
    return proc.stdout.strip()
