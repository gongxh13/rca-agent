import argparse
import csv
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
from shutil import copyfileobj
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Optional

from zoneinfo import ZoneInfo


SCENARIOS = ("bad_disk", "slow_disk", "pressure")


class CommandError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _fmt_utc_iso_seconds(ts_utc: datetime) -> str:
    return ts_utc.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fmt_utc_iso_millis(ts_utc: datetime) -> str:
    return ts_utc.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _fmt_window(start_utc: datetime, end_utc: datetime) -> str:
    return f"{_fmt_utc_iso_seconds(start_utc)}-{_fmt_utc_iso_seconds(end_utc)}"


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _which(cmd: str) -> Optional[str]:
    from shutil import which

    found = which(cmd)
    if found:
        return found
    for prefix in ("/usr/sbin", "/sbin", "/usr/local/sbin"):
        candidate = os.path.join(prefix, cmd)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _shell_quote(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    output_dir: Path
    record_csv: Path
    app_source: str
    syslog_source: str


@dataclass
class RunConfig:
    interval_seconds: int
    cycles: int
    local_tz: timezone
    dry_run: bool
    enable_noise: bool
    noise_mode: str
    output_layout: str
    noise_interval_seconds: float
    journalctl: Optional[str]
    logger: Optional[str]
    iptables: Optional[str]
    ping: Optional[str]
    fio: Optional[str]
    dmsetup: Optional[str]
    modprobe: Optional[str]
    mkfs: Optional[str]
    mount: Optional[str]
    umount: Optional[str]
    blockdev: Optional[str]
    truncate: Optional[str]
    dd: Optional[str]


def run_cmd(
    args: list[str],
    *,
    dry_run: bool,
    stdout: Optional[IO[str]] = None,
    stderr: Optional[IO[str]] = None,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        if stdout is not None:
            stdout.write(f"[dry-run] { _shell_quote(args) }\n")
            stdout.flush()
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
    return subprocess.run(args, stdout=stdout, stderr=stderr, check=check, text=text)


def popen_cmd(
    args: list[str],
    *,
    dry_run: bool,
    stdout: IO[str],
    stderr: Optional[IO[str]] = None,
) -> Optional[subprocess.Popen[str]]:
    if dry_run:
        stdout.write(f"[dry-run] { _shell_quote(args) }\n")
        stdout.flush()
        return None
    return subprocess.Popen(
        args,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )


def terminate_process(proc: Optional[subprocess.Popen[str]], timeout_seconds: float = 2.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return


def detect_syslog_source() -> str:
    for candidate in ("/var/log/messages", "/var/log/syslog"):
        if os.path.exists(candidate):
            return candidate
    return "/var/log/messages"


def parse_local_tz(tz_name: str) -> timezone:
    return ZoneInfo(tz_name)


def system_local_tz() -> timezone:
    return datetime.now().astimezone().tzinfo or timezone.utc


def find_scsi_debug_device(*, dry_run: bool, cmd_log: IO[str]) -> str:
    if dry_run:
        return "/dev/sda"

    def _is_scsi_debug(model: str, vendor: str) -> bool:
        m = (model or "").strip().lower()
        v = (vendor or "").strip().lower()
        if "scsi_debug" in m:
            return True
        if v == "linux" and "scsi" in m and "debug" in m:
            return True
        return False

    try:
        proc = subprocess.run(
            ["lsblk", "-J", "-o", "NAME,TYPE,VENDOR,MODEL"],
            check=True,
            text=True,
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

    try:
        proc = subprocess.run(
            ["lsblk", "-o", "NAME,TYPE,VENDOR,MODEL", "-nr"],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in proc.stdout.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 2:
                continue
            name = parts[0]
            type_ = parts[1]
            vendor = parts[2] if len(parts) >= 3 else ""
            model = parts[3] if len(parts) >= 4 else ""
            if type_ != "disk":
                continue
            if _is_scsi_debug(model, vendor):
                return f"/dev/{name}"
    except Exception:
        pass

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

    cmd_log.write("Failed to detect scsi_debug disk via lsblk/sysfs\n")
    cmd_log.flush()
    raise CommandError("cannot find scsi_debug device")


def ensure_modules(cfg: RunConfig, cmd_log: IO[str], modules: tuple[str, ...]) -> None:
    for module in modules:
        run_cmd(["sudo", cfg.modprobe or "modprobe", module], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)


def _is_mounted(mountpoint: str) -> bool:
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == mountpoint:
                    return True
    except OSError:
        return False
    return False


def cleanup_mount(mountpoint: str, cfg: RunConfig, cmd_log: IO[str]) -> None:
    if cfg.dry_run:
        run_cmd(["sudo", cfg.umount or "umount", "-f", mountpoint], dry_run=True, stdout=cmd_log, stderr=cmd_log, check=False)
        return
    if not _is_mounted(mountpoint):
        return
    run_cmd(["sudo", cfg.umount or "umount", "-f", mountpoint], dry_run=False, stdout=cmd_log, stderr=cmd_log, check=False)


def cleanup_scsi_debug(cfg: RunConfig, cmd_log: IO[str]) -> None:
    # Try to remove known consumers first (defensive cleanup)
    _dm_remove_if_exists("bad_disk", cfg, cmd_log)
    
    # Attempt 1
    import time
    time.sleep(0.5)
    proc = run_cmd(["sudo", cfg.modprobe or "modprobe", "-r", "scsi_debug"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    if proc.returncode == 0:
        return

    # Attempt 2 with more delay
    if not cfg.dry_run:
        cmd_log.write("Retrying scsi_debug cleanup...\n")
        cmd_log.flush()
        time.sleep(2.0)
        # Try to force remove if possible (usually not recommended, but here we just retry standard remove)
        run_cmd(["sudo", cfg.modprobe or "modprobe", "-r", "scsi_debug"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)

def _dm_mapping_exists(name: str, cfg: RunConfig) -> bool:
    if cfg.dry_run:
        return False
    try:
        proc = subprocess.run(
            ["sudo", cfg.dmsetup or "dmsetup", "info", name],
            check=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _dm_remove_if_exists(name: str, cfg: RunConfig, cmd_log: IO[str]) -> None:
    if cfg.dry_run:
        run_cmd(["sudo", cfg.dmsetup or "dmsetup", "remove", name], dry_run=True, stdout=cmd_log, stderr=cmd_log, check=False)
        return
    if not _dm_mapping_exists(name, cfg):
        return
    run_cmd(["sudo", cfg.dmsetup or "dmsetup", "remove", name], dry_run=False, stdout=cmd_log, stderr=cmd_log, check=False)


def bad_disk(cfg: RunConfig, cmd_log: IO[str]) -> None:
    ensure_modules(cfg, cmd_log, ("scsi_debug", "dm-error"))
    cleanup_mount("/mnt/bad", cfg, cmd_log)
    _dm_remove_if_exists("bad_disk", cfg, cmd_log)
    cleanup_scsi_debug(cfg, cmd_log)

    run_cmd(
        ["sudo", cfg.modprobe or "modprobe", "scsi_debug", "dev_size_mb=1024"],
        dry_run=cfg.dry_run,
        stdout=cmd_log,
        stderr=cmd_log,
        check=True,
    )
    device = find_scsi_debug_device(dry_run=cfg.dry_run, cmd_log=cmd_log)

    if cfg.dry_run:
        sectors = "0"
    else:
        proc = subprocess.run(["sudo", cfg.blockdev or "blockdev", "--getsz", device], check=True, text=True, capture_output=True)
        sectors = proc.stdout.strip()

    mapping = f"0 {sectors} error\n"
    if cfg.dry_run:
        cmd_log.write(f"[dry-run] sudo {cfg.dmsetup or 'dmsetup'} create bad_disk <<EOF\n{mapping}EOF\n")
        cmd_log.flush()
    else:
        proc = subprocess.run(
            ["sudo", cfg.dmsetup or "dmsetup", "create", "bad_disk"],
            input=mapping,
            text=True,
            stdout=cmd_log,
            stderr=cmd_log,
            check=True,
        )
        _ = proc

    run_cmd(["sudo", cfg.mkfs or "mkfs.ext4", "/dev/mapper/bad_disk"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    run_cmd(["sudo", "mkdir", "-p", "/mnt/bad"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    run_cmd(["sudo", cfg.mount or "mount", "/dev/mapper/bad_disk", "/mnt/bad"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    run_cmd(["sudo", cfg.dd or "dd", "if=/dev/zero", "of=/mnt/bad/file", "bs=4k", "count=100"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)

    cleanup_mount("/mnt/bad", cfg, cmd_log)
    _dm_remove_if_exists("bad_disk", cfg, cmd_log)
    cleanup_scsi_debug(cfg, cmd_log)


def slow_disk(cfg: RunConfig, cmd_log: IO[str]) -> None:
    ensure_modules(cfg, cmd_log, ("scsi_debug",))
    cleanup_mount("/mnt/slow", cfg, cmd_log)
    cleanup_scsi_debug(cfg, cmd_log)

    run_cmd(
        ["sudo", cfg.modprobe or "modprobe", "scsi_debug", "dev_size_mb=1024"],
        dry_run=cfg.dry_run,
        stdout=cmd_log,
        stderr=cmd_log,
        check=True,
    )
    device = find_scsi_debug_device(dry_run=cfg.dry_run, cmd_log=cmd_log)
    device_name = Path(device).name

    run_cmd(["sudo", cfg.mkfs or "mkfs.ext4", device], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    run_cmd(["sudo", "mkdir", "-p", "/mnt/slow"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    run_cmd(["sudo", cfg.mount or "mount", device, "/mnt/slow"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)

    run_cmd(
        ["sudo", "sh", "-c", "echo 1 > /sys/bus/pseudo/drivers/scsi_debug/every_nth"],
        dry_run=cfg.dry_run,
        stdout=cmd_log,
        stderr=cmd_log,
        check=False,
    )
    run_cmd(
        ["sudo", "sh", "-c", f"echo 30 > /sys/block/{device_name}/device/timeout"],
        dry_run=cfg.dry_run,
        stdout=cmd_log,
        stderr=cmd_log,
        check=False,
    )
    run_cmd(
        ["sudo", "sh", "-c", "echo 5000 > /sys/bus/pseudo/drivers/scsi_debug/delay"],
        dry_run=cfg.dry_run,
        stdout=cmd_log,
        stderr=cmd_log,
        check=False,
    )

    run_cmd(
        [
            cfg.fio or "fio",
            "--name=slow",
            "--filename=/mnt/slow/fio.dat",
            "--rw=randwrite",
            "--bs=4k",
            "--iodepth=16",
            "--direct=1",
            "--numjobs=2",
            "--size=200M",
            "--runtime=120",
            "--time_based",
        ],
        dry_run=cfg.dry_run,
        stdout=cmd_log,
        stderr=cmd_log,
        check=False,
    )

    cleanup_mount("/mnt/slow", cfg, cmd_log)
    cleanup_scsi_debug(cfg, cmd_log)


def _pressure_writer(stop: threading.Event, path: str, interval: float) -> None:
    i = 0
    while not stop.is_set():
        i += 1
        now = _fmt_utc_iso_seconds(_utc_now())
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{now} INFO handler succeed seq={i}\n")
        except OSError:
            pass
        stop.wait(interval)


def pressure(cfg: RunConfig, cmd_log: IO[str]) -> None:
    ensure_modules(cfg, cmd_log, ("scsi_debug",))
    cleanup_mount("/mnt/pressure", cfg, cmd_log)
    cleanup_scsi_debug(cfg, cmd_log)

    run_cmd(
        ["sudo", cfg.modprobe or "modprobe", "scsi_debug", "dev_size_mb=10240"],
        dry_run=cfg.dry_run,
        stdout=cmd_log,
        stderr=cmd_log,
        check=True,
    )
    device = find_scsi_debug_device(dry_run=cfg.dry_run, cmd_log=cmd_log)

    run_cmd(["sudo", cfg.mkfs or "mkfs.ext4", device], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    run_cmd(["sudo", "mkdir", "-p", "/mnt/pressure"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
    run_cmd(["sudo", cfg.mount or "mount", device, "/mnt/pressure"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)

    pressure_stop = threading.Event()
    pressure_thread = threading.Thread(target=_pressure_writer, args=(pressure_stop, "/mnt/pressure/app.log", 1.0), daemon=True)
    pressure_thread.start()
    try:
        run_cmd(["sudo", cfg.truncate or "truncate", "-s", "1G", "/mnt/pressure/file"], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)
        run_cmd(
            [
                cfg.fio or "fio",
                "--name=pressure",
                "--filename=/mnt/pressure/file",
                "--rw=write",
                "--bs=8k",
                "--iodepth=128",
                "--numjobs=4",
                "--runtime=120",
                "--time_based",
            ],
            dry_run=cfg.dry_run,
            stdout=cmd_log,
            stderr=cmd_log,
            check=False,
        )
    finally:
        pressure_stop.set()
        pressure_thread.join(timeout=2.0)
        cleanup_mount("/mnt/pressure", cfg, cmd_log)
        cleanup_scsi_debug(cfg, cmd_log)


SCENARIO_FUNCS = {
    "bad_disk": bad_disk,
    "slow_disk": slow_disk,
    "pressure": pressure,
}


def _required_bins_for_scenario(scenario: str) -> set[str]:
    base = {"modprobe", "umount", "mount", "mkfs.ext4", "fio", "truncate", "dd", "blockdev", "lsblk", "tail"}
    if scenario == "bad_disk":
        return base | {"dmsetup"}
    if scenario == "slow_disk":
        return base
    if scenario == "pressure":
        return base
    return base


def start_collectors(
    *,
    cfg: RunConfig,
    paths: Paths,
    tmp_dir: Path,
    cmd_log: IO[str],
) -> tuple[Optional[subprocess.Popen[str]], Optional[subprocess.Popen[str]], Optional[subprocess.Popen[str]]]:
    app_tmp = tmp_dir / "app.log"
    syslog_tmp = tmp_dir / "syslog.log"
    kernel_tmp = tmp_dir / "kernel.log"

    app_f = open(app_tmp, "w", encoding="utf-8")
    syslog_f = open(syslog_tmp, "w", encoding="utf-8")
    kernel_f = open(kernel_tmp, "w", encoding="utf-8")

    run_cmd(["sudo", "touch", paths.app_source], dry_run=cfg.dry_run, stdout=cmd_log, stderr=cmd_log, check=False)

    kernel_proc = None
    syslog_proc = None
    app_proc = None

    if cfg.journalctl:
        kernel_proc = popen_cmd(
            [cfg.journalctl, "-k", "-o", "short-iso-precise", "--utc", "-f"],
            dry_run=cfg.dry_run,
            stdout=kernel_f,
            stderr=cmd_log,
        )
        syslog_proc = popen_cmd(
            [cfg.journalctl, "-o", "short-iso-precise", "--utc", "-f"],
            dry_run=cfg.dry_run,
            stdout=syslog_f,
            stderr=cmd_log,
        )
    else:
        kernel_f.write("journalctl not found; kernel collector disabled\n")
        kernel_f.flush()
        syslog_proc = popen_cmd(["tail", "-n", "0", "-F", paths.syslog_source], dry_run=cfg.dry_run, stdout=syslog_f, stderr=cmd_log)

    app_proc = popen_cmd(["tail", "-n", "0", "-F", paths.app_source], dry_run=cfg.dry_run, stdout=app_f, stderr=cmd_log)

    app_f.close()
    syslog_f.close()
    kernel_f.close()

    return kernel_proc, syslog_proc, app_proc


SYSLOG_EN_TS_RE = re.compile(r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<hms>\d{2}:\d{2}:\d{2})(?P<rest>.*)$")
SYSLOG_CN_MD_RE = re.compile(r"^(?P<mon>\d{1,2})月\s+(?P<day>\d{1,2})\s+(?P<hms>\d{2}:\d{2}:\d{2})(?P<rest>.*)$")
CN_FULL_RE = re.compile(
    r"^(?P<year>\d{4})年\s*(?P<mon>\d{1,2})月\s*(?P<day>\d{1,2})日.*?(?P<hms>\d{2}:\d{2}:\d{2})\s*(?P<tz>[A-Za-z]{2,5})?(?P<rest>.*)$"
)
ISO_PREFIX_RE = re.compile(r"^(?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(?P<tz>Z|[+-]\d{2}:\d{2})?(?P<rest>.*)$")


def _month_to_int(mon: str) -> int:
    months = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    return months[mon]


def normalize_line_to_utc(line: str, *, base_year: int, local_tz: timezone) -> str:
    s = line.rstrip("\n")
    if not s:
        return line

    m = ISO_PREFIX_RE.match(s)
    if m:
        try:
            dt_str = m.group("iso") + (m.group("tz") or "")
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=local_tz)
            dt_utc = dt.astimezone(timezone.utc)
            prefix = _fmt_utc_iso_seconds(dt_utc)
            return f"{prefix}{m.group('rest')}\n"
        except Exception:
            return line

    m = SYSLOG_EN_TS_RE.match(s)
    if m:
        try:
            mon = _month_to_int(m.group("mon"))
            day = int(m.group("day"))
            hh, mm, ss = (int(x) for x in m.group("hms").split(":"))
            dt_local = datetime(base_year, mon, day, hh, mm, ss, tzinfo=local_tz)
            dt_utc = dt_local.astimezone(timezone.utc)
            prefix = _fmt_utc_iso_seconds(dt_utc)
            return f"{prefix}{m.group('rest')}\n"
        except Exception:
            return line

    m = SYSLOG_CN_MD_RE.match(s)
    if m:
        try:
            mon = int(m.group("mon"))
            day = int(m.group("day"))
            hh, mm, ss = (int(x) for x in m.group("hms").split(":"))
            dt_local = datetime(base_year, mon, day, hh, mm, ss, tzinfo=local_tz)
            dt_utc = dt_local.astimezone(timezone.utc)
            prefix = _fmt_utc_iso_seconds(dt_utc)
            return f"{prefix}{m.group('rest')}\n"
        except Exception:
            return line

    m = CN_FULL_RE.match(s)
    if m:
        try:
            year = int(m.group("year"))
            mon = int(m.group("mon"))
            day = int(m.group("day"))
            hh, mm, ss = (int(x) for x in m.group("hms").split(":"))
            tz_token = (m.group("tz") or "").strip()
            if tz_token == "CST":
                tz = ZoneInfo("Asia/Shanghai")
            else:
                tz = local_tz
            dt_local = datetime(year, mon, day, hh, mm, ss, tzinfo=tz)
            dt_utc = dt_local.astimezone(timezone.utc)
            prefix = _fmt_utc_iso_seconds(dt_utc)
            return f"{prefix}{m.group('rest')}\n"
        except Exception:
            return line

    return line


def normalize_file_in_place(path: Path, *, base_year: int, local_tz: timezone) -> None:
    tmp = path.with_suffix(path.suffix + ".normalized.tmp")
    with open(path, "r", encoding="utf-8", errors="replace") as r, open(tmp, "w", encoding="utf-8") as w:
        for line in r:
            w.write(normalize_line_to_utc(line, base_year=base_year, local_tz=local_tz))
    tmp.replace(path)


class NoiseEmitter:
    def __init__(self, *, cfg: RunConfig, app_source: str):
        self._cfg = cfg
        self._app_source = app_source
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self._cfg.enable_noise:
            return
        if self._cfg.noise_mode == "real":
            self._setup_kernel_log_hook()
        self._thread.start()

    def stop(self) -> None:
        if not self._cfg.enable_noise:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._cfg.noise_mode == "real":
            self._teardown_kernel_log_hook()

    def _emit_syslog(self, msg: str, *, priority: str) -> None:
        if not self._cfg.logger:
            return
        try:
            subprocess.run([self._cfg.logger, "-t", "rca-noise", "-p", priority, msg], check=False, text=True)
        except Exception:
            return

    def _setup_kernel_log_hook(self) -> None:
        if self._cfg.dry_run:
            return
        if not self._cfg.iptables:
            raise CommandError("noise_mode=real requires iptables")
        try:
            subprocess.run(["sudo", self._cfg.iptables, "-N", "RCA_NOISE"], check=False, text=True)
            subprocess.run(["sudo", self._cfg.iptables, "-F", "RCA_NOISE"], check=False, text=True)
            if subprocess.run(["sudo", self._cfg.iptables, "-C", "OUTPUT", "-j", "RCA_NOISE"], check=False).returncode != 0:
                subprocess.run(["sudo", self._cfg.iptables, "-I", "OUTPUT", "1", "-j", "RCA_NOISE"], check=False, text=True)
            subprocess.run(
                [
                    "sudo",
                    self._cfg.iptables,
                    "-A",
                    "RCA_NOISE",
                    "-p",
                    "icmp",
                    "-j",
                    "LOG",
                    "--log-prefix",
                    "rca-noise icmp: ",
                    "--log-level",
                    "6",
                ],
                check=False,
                text=True,
            )
            subprocess.run(["sudo", self._cfg.iptables, "-A", "RCA_NOISE", "-j", "RETURN"], check=False, text=True)
        except Exception:
            return

    def _teardown_kernel_log_hook(self) -> None:
        if self._cfg.dry_run:
            return
        if not self._cfg.iptables:
            return
        try:
            while subprocess.run(["sudo", self._cfg.iptables, "-D", "OUTPUT", "-j", "RCA_NOISE"], check=False).returncode == 0:
                pass
            subprocess.run(["sudo", self._cfg.iptables, "-F", "RCA_NOISE"], check=False, text=True)
            subprocess.run(["sudo", self._cfg.iptables, "-X", "RCA_NOISE"], check=False, text=True)
        except Exception:
            return

    def _emit_app(self, msg: str) -> None:
        if self._cfg.dry_run:
            return
        try:
            now = _fmt_utc_iso_seconds(_utc_now())
            with open(self._app_source, "a", encoding="utf-8") as f:
                f.write(f"{now} INFO {msg}\n")
        except OSError:
            return

    def _run(self) -> None:
        patterns = [
            "healthcheck ok service=api latency_ms={x}",
            "http request handled status=200 route=/v1/ping cost_ms={x}",
            "dns resolved name=example.com rtt_ms={x}",
            "worker heartbeat ok id={x}",
            "scheduler tick jobs={x}",
            "tls handshake ok peer=upstream-{x}",
        ]
        while not self._stop.is_set():
            x = random.randint(1, 500)
            msg = random.choice(patterns).format(x=x)
            if self._cfg.noise_mode == "logger":
                self._emit_syslog(msg, priority="user.info")
                self._emit_syslog(msg, priority="kern.info")
            elif self._cfg.noise_mode == "real":
                self._emit_kernel_event()
            self._emit_app(msg)
            self._stop.wait(self._cfg.noise_interval_seconds)

    def _emit_kernel_event(self) -> None:
        if self._cfg.dry_run:
            return
        if not self._cfg.ping:
            return
        try:
            subprocess.run([self._cfg.ping, "-c", "1", "-W", "1", "127.0.0.1"], check=False, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return


def append_record(
    record_csv: Path,
    *,
    start_utc: datetime,
    end_utc: datetime,
    fault_type: str,
    trigger_utc: datetime,
) -> None:
    exists = record_csv.exists()
    _safe_mkdir(record_csv.parent)
    with open(record_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["start_utc", "end_utc", "trigger_utc", "fault_type"])
        w.writerow(
            [
                _fmt_utc_iso_millis(start_utc),
                _fmt_utc_iso_millis(end_utc),
                _fmt_utc_iso_millis(trigger_utc),
                fault_type,
            ]
        )

def _day_dir_for_window(output_dir: Path, start_utc: datetime) -> Path:
    day = start_utc.astimezone(timezone.utc).date().isoformat()
    return output_dir / day


def _append_window_to_daily_log(*, dst: Path, src: Path, header: str) -> None:
    _safe_mkdir(dst.parent)
    with open(dst, "a", encoding="utf-8") as w:
        w.write(header)
        with open(src, "r", encoding="utf-8", errors="replace") as r:
            copyfileobj(r, w)
        if not header.endswith("\n"):
            w.write("\n")
        w.write("\n")


def run_cycle(
    *,
    cfg: RunConfig,
    paths: Paths,
    fault_type: str,
    start_utc: datetime,
    end_utc: datetime,
    stop_event: threading.Event,
) -> None:
    window = _fmt_window(start_utc, end_utc)
    tmp_dir = paths.output_dir / f"tmp_{window}"
    _safe_mkdir(tmp_dir)

    trigger_utc = start_utc
    max_delay_seconds = min(300.0, max(0.0, (end_utc - start_utc).total_seconds()))
    trigger_delay_seconds = random.uniform(0.0, max_delay_seconds) if max_delay_seconds > 0 else 0.0
    cmd_log_path = tmp_dir / "run.log"
    with open(cmd_log_path, "w", encoding="utf-8") as cmd_log:
        kernel_proc, syslog_proc, app_proc = start_collectors(cfg=cfg, paths=paths, tmp_dir=tmp_dir, cmd_log=cmd_log)
        try:
            cmd_log.write(f"window={window} fault_type={fault_type}\n")
            cmd_log.write(f"trigger_delay_seconds={trigger_delay_seconds:.3f}\n")
            cmd_log.flush()
            func = SCENARIO_FUNCS[fault_type]
            if trigger_delay_seconds > 0:
                stop_event.wait(trigger_delay_seconds)
            trigger_utc = _utc_now()
            func(cfg, cmd_log)
            remaining = (end_utc - _utc_now()).total_seconds()
            if remaining > 0:
                stop_event.wait(remaining)
        finally:
            terminate_process(kernel_proc)
            terminate_process(syslog_proc)
            terminate_process(app_proc)

    base_year = start_utc.astimezone(cfg.local_tz).year
    for name in ("app.log", "syslog.log", "kernel.log"):
        p = tmp_dir / name
        if p.exists():
            normalize_file_in_place(p, base_year=base_year, local_tz=cfg.local_tz)

    if cfg.output_layout == "daily":
        day_dir = _day_dir_for_window(paths.output_dir, start_utc)
        header = f"window={window} start_utc={_fmt_utc_iso_millis(start_utc)} end_utc={_fmt_utc_iso_millis(end_utc)} trigger_utc={_fmt_utc_iso_millis(trigger_utc)} fault_type={fault_type}\n"
        for name in ("app.log", "syslog.log", "kernel.log", "run.log"):
            src = tmp_dir / name
            if not src.exists():
                continue
            dst = day_dir / name
            _append_window_to_daily_log(dst=dst, src=src, header=header)
            src.unlink(missing_ok=True)
    else:
        for name in ("app.log", "syslog.log", "kernel.log", "run.log"):
            src = tmp_dir / name
            if not src.exists():
                continue
            dst = paths.output_dir / f"{window}_{name}"
            src.replace(dst)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    append_record(paths.record_csv, start_utc=start_utc, end_utc=end_utc, fault_type=fault_type, trigger_utc=trigger_utc)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="disk_fault_injector.py")
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--cycles", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=str(Path.cwd() / "fault_logs"))
    p.add_argument("--record-csv", type=str, default="")
    p.add_argument("--app-source", type=str, default="/var/log/app.log")
    p.add_argument("--syslog-source", type=str, default="")
    p.add_argument("--local-tz", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--noise-mode", type=str, default="logger", choices=["logger", "real"])
    p.add_argument("--output-layout", type=str, default="daily", choices=["daily", "window"])
    p.add_argument("--noise-interval-seconds", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", type=str, default="")
    return p


def main(argv: list[str]) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.seed:
        random.seed(args.seed)

    syslog_source = args.syslog_source or detect_syslog_source()
    local_tz = parse_local_tz(args.local_tz) if args.local_tz else system_local_tz()

    output_dir = Path(args.output_dir).resolve()
    _safe_mkdir(output_dir)

    record_csv = Path(args.record_csv).resolve() if args.record_csv else (output_dir / "fault_injection_record.csv")

    cfg = RunConfig(
        interval_seconds=args.interval_seconds,
        cycles=args.cycles,
        local_tz=local_tz,
        dry_run=args.dry_run,
        enable_noise=not args.no_noise,
        noise_mode=args.noise_mode,
        output_layout=args.output_layout,
        noise_interval_seconds=args.noise_interval_seconds,
        journalctl=_which("journalctl"),
        logger=_which("logger"),
        iptables=_which("iptables"),
        ping=_which("ping"),
        fio=_which("fio"),
        dmsetup=_which("dmsetup"),
        modprobe=_which("modprobe"),
        mkfs=_which("mkfs.ext4"),
        mount=_which("mount"),
        umount=_which("umount"),
        blockdev=_which("blockdev"),
        truncate=_which("truncate"),
        dd=_which("dd"),
    )

    paths = Paths(
        repo_root=Path(__file__).resolve().parent,
        output_dir=output_dir,
        record_csv=record_csv,
        app_source=args.app_source,
        syslog_source=syslog_source,
    )

    if args.scenario and args.scenario not in SCENARIOS:
        raise CommandError(f"unknown scenario: {args.scenario} (allowed: {SCENARIOS})")

    if not cfg.dry_run:
        scenarios = (args.scenario,) if args.scenario else SCENARIOS
        required = set().union(*(_required_bins_for_scenario(s) for s in scenarios))
        if cfg.enable_noise and cfg.noise_mode == "real":
            required |= {"iptables", "ping"}
        missing = [c for c in sorted(required) if _which(c) is None]
        if missing:
            raise CommandError(f"missing required commands: {', '.join(missing)}")

    stop_event = threading.Event()

    def _on_signal(_sig: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    noise = NoiseEmitter(cfg=cfg, app_source=paths.app_source)
    noise.start()
    try:
        cycle_idx = 0
        while not stop_event.is_set():
            cycle_idx += 1
            if cfg.cycles and cycle_idx > cfg.cycles:
                break
            start_utc = _utc_now()
            end_utc = start_utc + timedelta(seconds=cfg.interval_seconds)
            if args.scenario:
                fault_type = args.scenario
            else:
                fault_type = random.choice(SCENARIOS)
            if fault_type not in SCENARIOS:
                raise CommandError(f"unknown scenario: {fault_type} (allowed: {SCENARIOS})")
            run_cycle(
                cfg=cfg,
                paths=paths,
                fault_type=fault_type,
                start_utc=start_utc,
                end_utc=end_utc,
                stop_event=stop_event,
            )
    except CommandError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    finally:
        noise.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
