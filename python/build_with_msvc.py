"""
vcvarsall.bat の環境変数を Python プロセスに取り込み、uv build を実行するスクリプト。

日本語 Windows (cp932) では setuptools の cmd /u /c 方式が文字化けするため、
cmd /a /c (ANSI) で環境を取得して上書きする。
"""
import os
import subprocess
import sys


def get_msvc_env(arch: str = "x64") -> dict:
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Microsoft Visual Studio", "Installer", "vswhere.exe",
    )
    install_path = subprocess.check_output(
        [vswhere, "-latest", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath", "-products", "*"],
        stderr=subprocess.DEVNULL,
    ).decode("mbcs").strip()

    vcvarsall = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")

    # cmd /a = ANSI モード（cp932 として正しくデコードできる）
    out = subprocess.check_output(
        f'cmd /a /c ""{vcvarsall}" {arch} && set"',
        stderr=subprocess.STDOUT,
    ).decode("mbcs", errors="replace")

    env = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key and value:
            env[key] = value
    return env


def main():
    python_version = sys.argv[1] if len(sys.argv) > 1 else None

    print("[build_with_msvc] Loading MSVC environment via vcvarsall.bat (ANSI mode)...")
    msvc_env = get_msvc_env("x64")
    if "PATH" not in msvc_env:
        print("[build_with_msvc] ERROR: Failed to load MSVC environment.")
        sys.exit(1)

    # 現在の環境に MSVC 環境をマージ（MSVC のものを優先）
    merged = {**os.environ, **msvc_env}
    # CTRANSLATE2_ROOT は明示的に保持
    if "CTRANSLATE2_ROOT" in os.environ:
        merged["CTRANSLATE2_ROOT"] = os.environ["CTRANSLATE2_ROOT"]
    # setuptools に vcvarsall.bat の再実行をスキップさせ、この環境をそのまま使わせる
    merged["DISTUTILS_USE_SDK"] = "1"

    cmd = ["uv", "build", "--wheel"]
    if python_version:
        cmd += ["--python", python_version]
    else:
        cmd += ["--no-build-isolation"]

    print(f"[build_with_msvc] cl.exe: {msvc_env.get('PATH', '').split(';')[0]}")
    print(f"[build_with_msvc] Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        env=merged,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
