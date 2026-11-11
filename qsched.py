"""Memory-aware task queue for shared GPUs.

queue file lines:  name | mem_mb | gpus | deps | command
  mem_mb : GPU memory the task needs; it is launched on the first allowed GPU whose free memory (minus the declared
           memory of tasks launched there in the last SETTLE_S seconds, which may not have allocated yet) covers it
  gpus   : comma list of allowed GPU indices in order of preference, or '*' = every GPU, most free memory first
  deps   : comma list of task names that must have finished ok (or '-')
Lines starting with '#' are ignored.  The file is re-read every cycle, so tasks can be appended while it runs.
Failed tasks are retried up to QS_RETRIES times (default 3) after a cool-down.  Other users' processes are never
touched: the scheduler only reads nvidia-smi and launches our own commands.
State: logs/qsched_state.json;  per-task log: logs/<name>.log;  stage log: logs/stages_v40.log
"""
import json, os, subprocess, sys, time

ROOT = "/data/yaokang/evcache"
Q = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/queue.txt"
STATE = f"{ROOT}/logs/qsched_state.json"
STAGES = f"{ROOT}/logs/stages_v40.log"
MAX_PER_GPU = int(os.environ.get("QS_MAX", 3))
RETRIES = int(os.environ.get("QS_RETRIES", 3))
ALLOWED = [int(g) for g in os.environ["QS_GPUS"].split(",")] if os.environ.get("QS_GPUS") else None   # GPUs we may use at all
RESERVE_MB = int(os.environ.get("QS_RESERVE", 1200))      # head-room left on a GPU (other users' processes grow)
SETTLE_S = 180                                             # declared memory of a fresh task counts as used this long
COOLDOWN_S = 240                                           # wait before retrying a failed task


def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    free = {}
    for ln in out.strip().splitlines():
        i, u, t = [int(x) for x in ln.split(",")]
        if ALLOWED is None or i in ALLOWED:
            free[i] = t - u
    return free


def stage(msg):
    with open(STAGES, "a") as f:
        f.write(f"STAGE {msg} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def load_queue():
    tasks = []
    for ln in open(Q):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        name, mem, gpus, deps, cmd = [x.strip() for x in ln.split("|", 4)]
        tasks.append({"name": name, "mem": int(mem), "gpus": gpus, "deps": [] if deps == "-" else [d.strip() for d in deps.split(",")], "cmd": cmd})
    return tasks


def failed_log(name):
    """inspect only the output of the task's most recent launch"""
    try:
        txt = open(f"{ROOT}/logs/{name}.log", errors="ignore").read()
    except FileNotFoundError:
        return True
    seg = txt.rsplit("===== launch", 1)[-1][-8000:]
    return any(w in seg for w in ("Traceback", "Error:", "Killed", "OutOfMemoryError"))


state = json.load(open(STATE)) if os.path.exists(STATE) else {}
procs = {}
adopted = {n: s["pid"] for n, s in state.items() if s.get("status") == "running"}      # tasks of an earlier scheduler run
print("qsched start", time.strftime("%H:%M:%S"), "adopted", list(adopted), flush=True)
while True:
    tasks = load_queue()
    for name, p in list(procs.items()):
        rc = p.poll()
        if rc is not None:
            ok = rc == 0 and not failed_log(name)
            state[name]["status"] = "done" if ok else "fail"; state[name]["rc"] = rc; state[name]["end"] = time.time()
            stage(f"{'ok' if ok else 'FAIL'}_{name}")
            print("END", name, "rc", rc, "ok" if ok else "FAIL", time.strftime("%H:%M:%S"), flush=True)
            del procs[name]
    for name, pid in list(adopted.items()):
        if not os.path.exists(f"/proc/{pid}"):
            ok = not failed_log(name)
            state[name]["status"] = "done" if ok else "fail"; state[name]["rc"] = "adopted"; state[name]["end"] = time.time()
            stage(f"{'ok' if ok else 'FAIL'}_{name}")
            print("END(adopted)", name, "ok" if ok else "FAIL", time.strftime("%H:%M:%S"), flush=True)
            del adopted[name]
    free = gpu_free()
    now = time.time()
    running = {g: [n for n, s in state.items() if s.get("status") == "running" and s.get("gpu") == g and s.get("mem", 0) > 0] for g in free}
    for g in free:
        free[g] -= sum(state[n]["mem"] for n in running[g] if now - state[n]["start"] < SETTLE_S)
    launched = False
    for t in tasks:
        st = state.get(t["name"], {})
        if st.get("status") in ("running", "done"):
            continue
        if st.get("status") == "fail":
            if st.get("retries", 0) >= RETRIES or now - st.get("end", 0) < COOLDOWN_S:
                continue
        if any(state.get(d, {}).get("status") != "done" for d in t["deps"]):
            continue
        order = sorted(free, key=lambda g: -free[g]) if t["gpus"] == "*" else [int(g) for g in t["gpus"].split(",") if int(g) in free]
        for g in order:
            if g in free and free[g] - RESERVE_MB >= t["mem"] and len(running.get(g, [])) < MAX_PER_GPU:
                log = open(f"{ROOT}/logs/{t['name']}.log", "a")
                log.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu {g} =====\n"); log.flush()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                p = subprocess.Popen(["bash", "-c", t["cmd"]], stdout=log, stderr=subprocess.STDOUT, env=env,
                                     cwd=f"{ROOT}/code", start_new_session=True)
                procs[t["name"]] = p
                state[t["name"]] = {"status": "running", "gpu": g, "pid": p.pid, "start": now, "mem": t["mem"],
                                    "retries": st.get("retries", 0) + (1 if st.get("status") == "fail" else 0)}
                running.setdefault(g, []).append(t["name"]); free[g] -= t["mem"]
                stage(f"start_{t['name']}_gpu{g}")
                print("START", t["name"], "gpu", g, "pid", p.pid, "retry", state[t["name"]]["retries"], time.strftime("%H:%M:%S"), flush=True)
                launched = True
                break
        if launched:
            break                                   # one launch per cycle, then re-measure memory
    json.dump(state, open(STATE, "w"), indent=1)
    if os.path.exists(f"{ROOT}/qsched.stop"):
        print("stop file found", flush=True); break
    time.sleep(25 if launched else 40)
