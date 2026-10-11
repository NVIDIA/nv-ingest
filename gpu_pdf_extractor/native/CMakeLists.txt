cmake_minimum_required(VERSION 3.22)
project(gpu_pdfium LANGUAGES CXX CUDA)

# --- Phase-0 build skeleton ---------------------------------------------------
# Proves the full toolchain links into ONE Python-loadable module:
#   nanobind (Python bindings)  +  CUDA (sm_90 kernel)  +  PDFium (parser core).
# Real bindings land in P1; the CUDA rasterizer in P2.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)   # Hopper (sm_90), confirmed on this box
endif()

# Python (module-only components are enough for nanobind).
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind: discover its CMake package via the installed wheel.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# PDFium: prebuilt binaries shipped under third_party/pdfium (ships PDFiumConfig.cmake).
set(PDFium_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pdfium")
find_package(PDFium REQUIRED)

# P0 smoke module (nanobind+CUDA+PDFium link proof).
nanobind_add_module(gpu_pdfium_ext
  src/gpu_pdfium_ext.cpp
  src/cuda_kernel.cu)
target_include_directories(gpu_pdfium_ext PRIVATE "${PDFium_DIR}/include")
target_link_libraries(gpu_pdfium_ext PRIVATE pdfium)

# P1 native bindings module (pypdfium2-compatible drop-in core).
nanobind_add_module(_gpu_pdfium src/pdf_bindings.cpp)
target_include_directories(_gpu_pdfium PRIVATE "${PDFium_DIR}/include")
target_link_libraries(_gpu_pdfium PRIVATE pdfium)

# P2 CUDA rasterizer core.
nanobind_add_module(_gpu_raster src/raster_kernels.cu)

# So the loader finds libpdfium.so at import time without LD_LIBRARY_PATH gymnastics.
set_target_properties(gpu_pdfium_ext _gpu_pdfium PROPERTIES
  BUILD_RPATH "${PDFium_DIR}/lib"
  INSTALL_RPATH "${PDFium_DIR}/lib")
