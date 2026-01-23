#!/usr/bin/env python3
import os
import shutil
import argparse
import paramiko
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
import logging

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.config.loader import load_config

# Configure logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("fetch_logs")
console = Console()

# Default configuration
# Default file mapping: destination -> source
DEFAULT_FILE_MAPPING = {
    "app.log": "app.log",
    "kernel.log": "kernel.log",
    "syslog.log": "/var/log/syslog"
}
DATASET_ROOT = Path("datasets/disk_fault_logs")

def fetch_local_logs(src_dir: str, target_dir: Path, files_map: dict[str, str]):
    """Fetch logs from local directory or absolute paths."""
    logger.info(f"Fetching logs locally (Base dir: {src_dir})")
    base_path = Path(src_dir)
    
    success_count = 0
    for dest_name, src_pattern in files_map.items():
        # Determine source path
        if os.path.isabs(src_pattern):
            src_file = Path(src_pattern)
        else:
            src_file = base_path / src_pattern

        if src_file.exists():
            try:
                shutil.copy2(src_file, target_dir / dest_name)
                logger.info(f"Copied {src_file} -> {dest_name}")
                success_count += 1
            except PermissionError:
                logger.error(f"Permission denied reading {src_file}. Try running with sudo.")
            except Exception as e:
                logger.error(f"Error copying {src_file}: {e}")
        else:
            logger.warning(f"Log file not found: {src_file}")
    
    return success_count > 0

def fetch_remote_logs(host, port, user, password, remote_base_dir, target_dir: Path, files_map: dict[str, str]):
    """Fetch logs from remote server via SSH/SFTP with compression."""
    logger.info(f"Fetching logs from remote server: {user}@{host} (Base dir: {remote_base_dir})")
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=user, password=password)
        
        # 1. Create remote temp directory
        stdin, stdout, stderr = ssh.exec_command("mktemp -d")
        remote_tmp_dir = stdout.read().decode().strip()
        if not remote_tmp_dir:
            raise Exception("Failed to create remote temp directory")
        
        logger.info(f"Created remote temp dir: {remote_tmp_dir}")
        
        # 2. Copy files to temp dir with target names
        files_found = False
        for dest_name, src_pattern in files_map.items():
            if os.path.isabs(src_pattern):
                remote_src = src_pattern
            else:
                remote_src = str(Path(remote_base_dir) / src_pattern).replace("\\", "/")
            
            remote_dest = f"{remote_tmp_dir}/{dest_name}"
            
            # Try normal copy, fallback to sudo -n copy
            cmd = f"cp '{remote_src}' '{remote_dest}' 2>/dev/null || sudo -n cp '{remote_src}' '{remote_dest}'"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status == 0:
                # Ensure readability for the archive user (current user)
                # If we used sudo to copy, the file might belong to root.
                # We try to chown it back to the current user, or make it world readable.
                # Simplest is chmod a+r
                ssh.exec_command(f"sudo -n chmod a+r '{remote_dest}' || chmod a+r '{remote_dest}'")
                
                logger.info(f"Prepared remote file: {dest_name}")
                files_found = True
            else:
                # If both failed, try to read stderr from the second attempt if possible, 
                # but we redirected first one. 
                # Just log a general warning.
                logger.warning(f"Failed to copy {remote_src} (Permission denied or file not found)")

        if not files_found:
            logger.warning("No log files found on remote server.")
            ssh.exec_command(f"rm -rf {remote_tmp_dir}")
            ssh.close()
            return False

        # 3. Compress files
        remote_archive = f"{remote_tmp_dir}/logs_archive.tar.gz"
        # tar -czf <archive> -C <dir> .
        # Exclude the archive itself if it's in the same dir (it shouldn't be matched by . but just in case)
        cmd = f"cd '{remote_tmp_dir}' && tar -czf logs_archive.tar.gz *"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        if stdout.channel.recv_exit_status() != 0:
            err = stderr.read().decode()
            raise Exception(f"Remote compression failed: {err}")
        
        logger.info("Remote logs compressed successfully")

        # 4. Download archive
        sftp = ssh.open_sftp()
        local_archive = target_dir / "logs_archive.tar.gz"
        try:
            sftp.get(remote_archive, str(local_archive))
            logger.info(f"Downloaded archive to {local_archive}")
        finally:
            sftp.close()

        # 5. Clean up remote
        ssh.exec_command(f"rm -rf {remote_tmp_dir}")
        ssh.close()

        # 6. Extract local archive
        import tarfile
        try:
            with tarfile.open(local_archive, "r:gz") as tar:
                # Avoid extraction outside target directory (security best practice)
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory
                
                safe_members = []
                for member in tar.getmembers():
                    member_path = os.path.join(str(target_dir), member.name)
                    if is_within_directory(str(target_dir), member_path):
                        safe_members.append(member)
                
                tar.extractall(path=target_dir, members=safe_members)
            
            logger.info("Logs extracted successfully")
        finally:
            # 7. Remove local archive
            if local_archive.exists():
                os.remove(local_archive)
        
        return True

    except Exception as e:
        logger.error(f"Error fetching remote logs: {e}")
        try:
            if 'ssh' in locals():
                ssh.close()
        except:
            pass
        return False

def run_fetch_logs(mode, local_dir, host, port, user, password, remote_dir, date_str, files_map):
    # 1. Prepare target directory
    target_date_dir = DATASET_ROOT / date_str
    if target_date_dir.exists():
        logger.warning(f"Target directory exists, cleaning up: {target_date_dir}")
        shutil.rmtree(target_date_dir)
    
    target_date_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created target directory: {target_date_dir}")
    
    # 2. Fetch logs
    success = False
    if mode == "local":
        success = fetch_local_logs(local_dir, target_date_dir, files_map)
    else:
        if not all([host, user, password]):
            logger.error("Remote mode requires host, user, and password (via CLI or config.yaml)")
            return False
        success = fetch_remote_logs(host, port, user, password, remote_dir, target_date_dir, files_map)
    
    if success:
        logger.info(f"Logs successfully fetched to {target_date_dir}")
    else:
        logger.error("Failed to fetch any logs.")
        # Cleanup empty dir if failed
        if target_date_dir.exists() and not any(target_date_dir.iterdir()):
             target_date_dir.rmdir()
    
    return success

def main():
    # Load config
    try:
        full_config = load_config()
        cfg = full_config.log_fetch
    except Exception as e:
        logger.debug(f"Config load failed or empty: {e}")
        cfg = None

    parser = argparse.ArgumentParser(description="Fetch and archive fault logs.")
    
    # Mode selection
    parser.add_argument("--mode", choices=["local", "remote"], help="Fetch mode: local or remote")
    
    # Local config
    parser.add_argument("--local-dir", help="Source directory for local logs")
    
    # Remote config
    parser.add_argument("--host", help="Remote host IP")
    parser.add_argument("--port", type=int, help="Remote SSH port")
    parser.add_argument("--user", help="Remote username")
    parser.add_argument("--password", help="Remote password")
    parser.add_argument("--remote-dir", help="Remote log directory")
    
    # Date config
    parser.add_argument("--date", help="Target date (YYYY-MM-DD), defaults to today", default=datetime.now().strftime("%Y-%m-%d"))
    
    args = parser.parse_args()
    
    # Resolve configuration (CLI args > Config file > Defaults)
    mode = args.mode or (cfg.mode if cfg else "local")
    
    local_dir = args.local_dir or (cfg.local_source_dir if cfg else ".")
    
    host = args.host or (cfg.remote_ip if cfg else None)
    port = args.port or (cfg.remote_port if cfg else 22)
    user = args.user or (cfg.remote_username if cfg else None)
    password = args.password or (cfg.remote_password if cfg else None)
    remote_dir = args.remote_dir or (cfg.remote_log_dir if cfg else "/tmp/fault_logs")
    
    # Resolve file mapping
    files_map = cfg.files if (cfg and cfg.files) else DEFAULT_FILE_MAPPING

    run_fetch_logs(
        mode=mode,
        local_dir=local_dir,
        host=host,
        port=port,
        user=user,
        password=password,
        remote_dir=remote_dir,
        date_str=args.date,
        files_map=files_map
    )

if __name__ == "__main__":
    main()
