"""VoltSnap 环境验证脚本"""
import sys
import subprocess
import importlib


def check_python_version():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 11
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Python {v.major}.{v.minor}.{v.micro}")
    return ok


def check_import(module_name, package_name=None):
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[PASS] {package_name or module_name} ({ver})")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name or module_name}: {e}")
        return False


def check_ximgproc():
    try:
        import cv2

        thin = cv2.ximgproc.thinning
        print("[PASS] cv2.ximgproc.thinning available")
        return True
    except (ImportError, AttributeError) as e:
        print(f"[FAIL] cv2.ximgproc.thinning: {e}")
        return False


def check_ngspice():
    try:
        r = subprocess.run(
            ["ngspice", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        ver = r.stdout.strip() or r.stderr.strip()
        print(f"[PASS] ngspice: {ver}")
        return True
    except FileNotFoundError:
        print("[FAIL] ngspice: not found in PATH")
        return False
    except Exception as e:
        print(f"[FAIL] ngspice: {e}")
        return False


def main():
    print("=== VoltSnap Environment Verification ===\n")

    results = []
    results.append(check_python_version())
    results.append(check_import("cv2", "opencv-contrib-python"))
    results.append(check_import("numpy"))
    results.append(check_import("matplotlib"))
    results.append(check_import("networkx"))
    results.append(check_import("schemdraw"))
    results.append(check_import("yaml", "pyyaml"))
    results.append(check_import("pytest"))
    results.append(check_ximgproc())
    results.append(check_ngspice())

    passed = sum(results)
    total = len(results)
    print(f"\n=== {passed}/{total} checks passed ===")

    if passed == total:
        print("Environment is ready!")
    else:
        print("Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
