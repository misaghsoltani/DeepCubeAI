#!/usr/bin/env bash

# Determine number of usable cores: prefer Slurm env, otherwise use system CPU count
# Result: NUM_CORES = (allocated_cores - 1), but at least 1.

get_allocated_cpus() {
    # 1) Slurm: prefer SLURM_CPUS_ON_NODE if numeric
    if [[ -n "${SLURM_CPUS_ON_NODE:-}" && "${SLURM_CPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
        echo "${SLURM_CPUS_ON_NODE}"
        return
    fi
    
    # 2) Slurm: parse the first number from SLURM_JOB_CPUS_PER_NODE (e.g., "16(x2),8")
    if [[ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]]; then
        if [[ "${SLURM_JOB_CPUS_PER_NODE}" =~ ^([0-9]+) ]]; then
            echo "${BASH_REMATCH[1]}"
            return
        fi
    fi
    
    # 3) System detection (platform-independent fallbacks)
    if command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN && return
    fi
    if command -v nproc >/dev/null 2>&1; then
        nproc && return
    fi
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu && return
    fi
    if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
        return
    fi
    
    # 4) Last resort
    echo 1
}
