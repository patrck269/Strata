"""Start llama-server with no console. llama-swap owns this process."""
import ctypes
import subprocess
import sys
from ctypes import wintypes

CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9

SERVER = r"M:\Strata\llama-cpp\llama-server.exe"
MODEL = r"M:\Strata-data\models\qwen3.8-27b\Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf"
# Vision projector stays on the CPU so the GPU keeps its room for weights and the two KV slots.
MMPROJ = r"M:\Strata-data\models\qwen3.8-27b\mmproj-model-bf16.gguf"


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def bind_kill_on_close(proc):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    kernel32.AssignProcessToJobObject(job, int(proc._handle))
    proc._job = job


def main():
    if len(sys.argv) != 2:
        print("usage: llama-qwen38-27b.py PORT", file=sys.stderr)
        return 2
    cmd = [
        SERVER,
        "-m", MODEL,
        "--mmproj", MMPROJ,
        "--no-mmproj-offload",
        "-c", "65536",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "-fa", "on",
        "-ngl", "99",
        "-np", "2",
        "--no-kv-unified",
        "--fit", "off",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "2",
        "--jinja",
        "--host", "127.0.0.1",
        "--port", sys.argv[1],
    ]
    proc = subprocess.Popen(cmd, creationflags=CREATE_NO_WINDOW)
    bind_kill_on_close(proc)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
