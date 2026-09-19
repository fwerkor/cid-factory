from __future__ import annotations

import csv
import io
import subprocess

from .models import HardwareDevice


def _number(value: str) -> float | None:
    value = value.strip()
    if not value or value.casefold() in {"n/a", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def cuda_devices() -> list[HardwareDevice]:
    fields = [
        "index",
        "name",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices: list[HardwareDevice] = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != len(fields):
            continue
        devices.append(
            HardwareDevice(
                index=int(row[0]),
                name=row[1].strip(),
                kind="cuda",
                memory_total_mb=_number(row[2]),
                memory_used_mb=_number(row[3]),
                utilization_percent=_number(row[4]),
                temperature_c=_number(row[5]),
                power_w=_number(row[6]),
            )
        )
    return devices
