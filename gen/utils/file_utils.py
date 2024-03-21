import shutil
from pathlib import Path
import subprocess

import sysrsync

from gen import SCRATCH_CACHE_PATH
from gen.utils.logging_utils import log_info, log_warn

def get_available_path(nfs_path: Path, scratch_prefix: Path = SCRATCH_CACHE_PATH, return_scratch_only: bool = False, return_scratch_path_if_exists: bool = True, resolve: bool = True) -> Path:
    nfs_path = Path(nfs_path)

    if resolve:
        nfs_path = nfs_path.resolve()
        scratch_prefix = scratch_prefix.resolve()

    scratch_path = scratch_prefix / nfs_path.relative_to(nfs_path.anchor)

    if return_scratch_path_if_exists and scratch_path.exists():
        return scratch_path
    
    if return_scratch_only: scratch_path.parent.mkdir(parents=True, exist_ok=True)
    return scratch_path if return_scratch_only else nfs_path


def sync_data(nfs_path: Path, scratch_prefix: Path = SCRATCH_CACHE_PATH, sync: bool = True, run_space_check: bool = True, **kwargs) -> None:
    nfs_path.mkdir(parents=True, exist_ok=True)
    scratch_path = get_available_path(nfs_path, scratch_prefix, return_scratch_only=True)

    if sync or not scratch_path.exists():
        if not sync:
            log_warn(f"Scratch path {scratch_path} does not exist, syncing data from NFS")

        scratch_path.mkdir(parents=True, exist_ok=True)

        if run_space_check:
            _, _, free = shutil.disk_usage(scratch_prefix.parents[1])
            nfs_size = sum(f.stat().st_size for f in nfs_path.glob('**/*') if f.is_file())

            if free < nfs_size:
                raise RuntimeError(f"Not enough space on {scratch_prefix.parents[1]}. Need {nfs_size} bytes, have {free} bytes free.")

        log_info(f"Rsync[ing] data from {nfs_path} to {scratch_path}...")

        if len(kwargs) == 0:
            sysrsync.run(source=str(nfs_path), destination=str(scratch_path), options=["--archive", "--progress"])
        else:
            cmd = sysrsync.get_rsync_command(source=str(nfs_path), destination=str(scratch_path), **kwargs)
            log_info(f"Running rsync command: {cmd}")
            subprocess.run(cmd)

    return scratch_path