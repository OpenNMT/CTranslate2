import os

import ctranslate2


class _FakeNvidiaModule:
    def __init__(self, path):
        self.__path__ = [path]


def _make_import_module(mapping):
    def _import_module(name):
        if name not in mapping:
            raise ImportError(name)
        return mapping[name]

    return _import_module


def test_registers_bin_dirs_for_installed_nvidia_packages(tmp_path, monkeypatch):
    cublas_dir = tmp_path / "nvidia" / "cublas"
    cudnn_dir = tmp_path / "nvidia" / "cudnn"
    cublas_bin = cublas_dir / "bin"
    cudnn_bin = cudnn_dir / "bin"
    cublas_bin.mkdir(parents=True)
    cudnn_bin.mkdir(parents=True)

    import_module = _make_import_module(
        {
            "nvidia.cublas": _FakeNvidiaModule(str(cublas_dir)),
            "nvidia.cudnn": _FakeNvidiaModule(str(cudnn_dir)),
        }
    )

    registered = []
    monkeypatch.setattr(
        os, "add_dll_directory", lambda p: registered.append(p), raising=False
    )
    monkeypatch.setattr(os, "environ", {"PATH": "C:\\existing"})

    ctranslate2._register_nvidia_pip_dll_directories(import_module=import_module)

    assert registered == [str(cublas_bin), str(cudnn_bin)]
    assert (
        os.environ["PATH"]
        == str(cudnn_bin) + os.pathsep + str(cublas_bin) + os.pathsep + "C:\\existing"
    )


def test_skips_package_that_is_not_pip_installed(monkeypatch):
    import_module = _make_import_module({})  # neither package importable

    registered = []
    monkeypatch.setattr(
        os, "add_dll_directory", lambda p: registered.append(p), raising=False
    )
    original_path = dict(os.environ)

    ctranslate2._register_nvidia_pip_dll_directories(import_module=import_module)

    assert registered == []
    assert dict(os.environ) == original_path


def test_skips_when_bin_subdir_does_not_exist(tmp_path, monkeypatch):
    package_dir = tmp_path / "nvidia" / "cublas"
    package_dir.mkdir(parents=True)  # deliberately no "bin" subdir inside

    import_module = _make_import_module(
        {"nvidia.cublas": _FakeNvidiaModule(str(package_dir))}
    )

    registered = []
    monkeypatch.setattr(
        os, "add_dll_directory", lambda p: registered.append(p), raising=False
    )
    original_path = dict(os.environ)

    ctranslate2._register_nvidia_pip_dll_directories(import_module=import_module)

    assert registered == []
    assert dict(os.environ) == original_path


def test_path_still_updated_when_add_dll_directory_raises_oserror(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "nvidia" / "cublas"
    bin_dir = package_dir / "bin"
    bin_dir.mkdir(parents=True)

    import_module = _make_import_module(
        {"nvidia.cublas": _FakeNvidiaModule(str(package_dir))}
    )

    def _raise(_path):
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "add_dll_directory", _raise, raising=False)
    monkeypatch.setattr(os, "environ", {"PATH": "C:\\existing"})

    # Must not raise, even though add_dll_directory() fails.
    ctranslate2._register_nvidia_pip_dll_directories(import_module=import_module)

    assert os.environ["PATH"] == str(bin_dir) + os.pathsep + "C:\\existing"
