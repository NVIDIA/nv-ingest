# gpu_pdfium native (Phase-0 build skeleton)

Proves the toolchain links **nanobind + CUDA (sm_90) + PDFium** into one Python-loadable module.
Real bindings (P1) and the CUDA rasterizer (P2) build on this.

## Layout
- `CMakeLists.txt` — CXX+CUDA project; finds Python/nanobind/PDFium.
- `src/gpu_pdfium_ext.cpp` — nanobind module (`cuda_add`, `cuda_device_info`, `pdfium_init_check`).
- `src/cuda_kernel.cu` — trivial sm_90 kernel (placeholder for the rasterizer).
- `third_party/pdfium/` — prebuilt PDFium 151.0.7906.0 (libpdfium.so + headers + PDFiumConfig.cmake).
- `pyproject.toml` — scikit-build-core config for the future wheel.

## Build & test
```bash
. /etc/profile.d/cuda.sh        # puts CUDA 13.0 nvcc on PATH
cd gpu_pdf_extractor/native
cmake -S . -B build -G Ninja \
  -DPython_EXECUTABLE=$(which python3) \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
PYTHONPATH=build python3 -c "import gpu_pdfium_ext as g; \
  print(g.cuda_add(2,40), g.cuda_device_info(), g.pdfium_init_check())"
```
Expected: `42 cuda devices=8 dev0=NVIDIA H100 NVL sm_90 pdfium init ok (last_error=0)`

## Notes / P1 follow-ups
- PDFium prebuilt is build **7906**; pypdfium2 (the baseline) is build **6462**. Match versions in P1
  for faithful parity, or pin the baseline to whatever PDFium we link.
- To regenerate the prebuilt PDFium:
  `curl -L .../pdfium-linux-x64.tgz | tar xz -C third_party/pdfium`
  (from github.com/bblanchon/pdfium-binaries/releases/latest).
- `cmake.version >=3.22` works; upgrade to >=3.27 if scikit-build-core wheel builds need it.
