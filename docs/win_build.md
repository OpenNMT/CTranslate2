# Windows でのビルド手順

このドキュメントでは、Windows 環境において CTranslate2 の C++ ライブラリをビルドし、その後 Python の wheel パッケージを作成するまでの手順を説明します。

---

## 前提条件

以下のツール・ライブラリを事前にインストールしてください。

| ツール | 最低バージョン | 備考 |
|--------|--------------|------|
| Visual Studio | 2019 以降 | 「C++ によるデスクトップ開発」ワークロードが必要 |
| CMake | 3.15 以降 | [cmake.org](https://cmake.org/download/) からインストール |
| Python | 3.9 以降 | |
| Intel oneAPI MKL | 2019.5 以降 | CPU バックエンドとして使用（デフォルト）。[Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) に含まれる |
| CUDA Toolkit | 11.0 以降 | GPU サポートが必要な場合のみ |
| cuDNN | 8 以降 | 畳み込みモデル（音声認識等）を使用する場合のみ |

> **Note:** 以降のコマンドはすべて **x64 Native Tools Command Prompt for VS 2019**（または VS 2022）で実行してください。
> [スタートメニュー] → [Visual Studio 20xx] → [x64 Native Tools Command Prompt for VS 20xx]

---

## 1. ソースコードの取得

```cmd
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
```

サブモジュールを含めてクローンするため `--recursive` が必要です。

---

## 2. C++ ライブラリのビルド

### 2-1. ビルドディレクトリを作成して CMake を実行

**CPU のみ（Intel MKL バックエンド）:**

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%CD%\..\install" "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

Visual Studio 2022 を使用する場合は `-G "Visual Studio 17 2022"` に変更してください。

**CPU + GPU（CUDA バックエンド）:**

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%CD%\..\install" ^
    -DWITH_CUDA=ON ^
    -DWITH_CUDNN=ON ^
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

### 2-2. ビルドとインストール

```cmd
cmake --build . --config Release --parallel
cmake --install . --config Release
```

成功すると `install\` ディレクトリ以下に以下が生成されます。

```
install\
  bin\
    ctranslate2_translator.exe   ← CLI ツール
  include\
    ctranslate2\                 ← C++ ヘッダー
  lib\
    ctranslate2.lib              ← インポートライブラリ
  bin\
    ctranslate2.dll              ← 共有ライブラリ（wheel のビルドで必要）
```

> **Tip:** `install\bin` を `PATH` に追加しておくと、Python ラッパーが実行時に DLL を見つけやすくなります。

---

## 3. Python wheel のビルド

### 3-1. 依存 DLL をパッケージディレクトリにコピー

`ctranslate2.dll` は `libiomp5md.dll`（Intel OpenMP ランタイム）に依存しています。
wheel に同梱するため、ビルド前にパッケージディレクトリへコピーします。

```cmd
copy "%CTRANSLATE2_ROOT%\bin\ctranslate2.dll" python\ctranslate2\
copy "%ONEAPI_ROOT%compiler\latest\bin\libiomp5md.dll" python\ctranslate2\
```

`ONEAPI_ROOT` が未設定の場合は `%ProgramFiles(x86)%\Intel\oneAPI\` を使用してください。

### 3-2. ビルド依存パッケージのインストール

`uv` を使ってビルドに必要なパッケージをインストールします。

```cmd
cd python
uv pip install --system setuptools wheel pybind11==2.11.1
```

### 3-3. wheel のビルド

`build_with_msvc.py` スクリプトが `vcvarsall.bat` の環境を自動で取り込み、
`uv build --wheel --no-build-isolation` を実行します。

```cmd
set CTRANSLATE2_ROOT=<C++ライブラリのインストールパス>
python build_with_msvc.py
```

> **Note:** `python setup.py bdist_wheel`（旧来の方法）の代わりに `uv build` を使っています。
> `--no-build-isolation` は MSVC コンパイラ環境を現在のシェルから継承するために必要です。

ビルドが完了すると `dist\` ディレクトリに `.whl` ファイルが生成されます。

```
python\
  dist\
    ctranslate2-X.Y.Z-cpXX-cpXX-win_amd64.whl
```

### 3-4. wheel のインストール

```cmd
uv pip install --system dist\ctranslate2-*.whl
```

---

## 4. 動作確認

```cmd
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

バージョン番号が表示されれば正常にインストールされています。

---

## 5. よくあるエラーと対処

| エラー | 原因 | 対処 |
|--------|------|------|
| `MKL not found` | MKLROOT 環境変数が未設定 | Intel oneAPI の `setvars.bat` を実行してから cmake を再実行 |
| `CUDA not found` | CUDA_PATH 環境変数が未設定 | CUDA Toolkit の再インストール、または `-DCUDA_TOOLKIT_ROOT_DIR=<path>` を追加 |
| `DLL not found` 実行時エラー | `install\bin` が PATH に未追加 | `set PATH=%CTRANSLATE2_ROOT%\bin;%PATH%` を実行 |
| `cl.exe not found` | x64 Native Tools Prompt を使用していない | x64 Native Tools Command Prompt から再実行 |
| `Invalid CMAKE_POLICY_VERSION_MINIMUM value "3"` | `3.5` が `3` と `.5` に分割される | `-D` オプション全体をダブルクォートで囲む: `"-DCMAKE_POLICY_VERSION_MINIMUM=3.5"` |
| `Unable to find a compatible Visual Studio installation` | 日本語 Windows で `vcvarsall.bat` の出力が UTF-16LE として正しくデコードできない | `build_with_msvc.py` を使用してビルドする（ANSI モードで環境を取得） |
| `Could not find module 'ctranslate2.dll' (or one of its dependencies)` | `libiomp5md.dll` が wheel に含まれていない | `libiomp5md.dll` を `python\ctranslate2\` にコピーしてから再ビルド |

---

## 参考

- [Build options 一覧](installation.md#build-options)
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
