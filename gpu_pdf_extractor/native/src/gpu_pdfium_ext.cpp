// Phase-0 nanobind smoke module: links nanobind + CUDA + PDFium in one extension.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include "fpdfview.h"

namespace nb = nanobind;

// Defined in cuda_kernel.cu (compiled by nvcc, linked into this module).
int cuda_add(int a, int b);
std::string cuda_device_info();

// Proves libpdfium.so links and initializes at runtime from inside the module.
static std::string pdfium_init_check() {
  FPDF_LIBRARY_CONFIG cfg{};
  cfg.version = 2;
  cfg.m_pUserFontPaths = nullptr;
  cfg.m_pIsolate = nullptr;
  cfg.m_v8EmbedderSlot = 0;
  FPDF_InitLibraryWithConfig(&cfg);
  unsigned long err = FPDF_GetLastError();
  FPDF_DestroyLibrary();
  return "pdfium init ok (last_error=" + std::to_string(err) + ")";
}

NB_MODULE(gpu_pdfium_ext, m) {
  m.doc() = "gpu_pdfium Phase-0 skeleton: nanobind + CUDA + PDFium link proof";
  m.def("cuda_add", &cuda_add, "GPU a+b", nb::arg("a"), nb::arg("b"));
  m.def("cuda_device_info", &cuda_device_info, "CUDA device summary string");
  m.def("pdfium_init_check", &pdfium_init_check, "Initialize+teardown PDFium; proves it links");
}
