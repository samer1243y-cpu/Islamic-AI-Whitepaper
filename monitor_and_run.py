import os
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "indexes"
EMBED_FILE = INDEX_DIR / "embeddings.npy"
FAISS_FILE = INDEX_DIR / "faiss.index"
METADATA = INDEX_DIR / "metadata.json"

CHECK_INTERVAL = 10
STABLE_ROUNDS = 3

def wait_for_files():
    print("Monitor: waiting for index files to appear...")
    while not (EMBED_FILE.exists() and FAISS_FILE.exists()):
        time.sleep(CHECK_INTERVAL)
    print("Monitor: index files detected.")

def wait_for_stable_files():
    print("Monitor: waiting for files to become stable...")
    stable = 0
    last_size = -1
    while stable < STABLE_ROUNDS:
        try:
            size = EMBED_FILE.stat().st_size + FAISS_FILE.stat().st_size
        except Exception:
            size = -1
        if size == last_size and size > 0:
            stable += 1
            print(f"Monitor: stable round {stable}/{STABLE_ROUNDS} (size={size})")
        else:
            stable = 0
            print(f"Monitor: size changed or initializing (size={size}), resetting stability counter")
        last_size = size
        time.sleep(CHECK_INTERVAL)
    print("Monitor: files appear stable. Proceeding to run retrieval tests.")


def run_tests():
    env = os.environ.copy()
    env.setdefault("ENABLE_RERANKER", "false")
    cmd = [env.get("PYTHON", "python"), "-c", "import runpy; runpy.run_path('scripts/run_retrieval_tests.py', run_name='__main__')"]
    print("Monitor: running retrieval tests...")
    proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print(f"Monitor: retrieval tests exited with code {proc.returncode}")


def main():
    wait_for_files()
    wait_for_stable_files()
    run_tests()

if __name__ == '__main__':
    main()
