// P2 — CUDA batched anti-aliased polygon-fill rasterizer (core primitive).
//
// Scope: the central rasterization op — fill arbitrarily-shaped polygons with anti-aliasing,
// composited in painter's order over a white background, batched across many pages and GPUs.
// This is the data-parallel heart of a page rasterizer. NOT yet a full PDFium replacement
// (no glyph hinting / clipping / blend modes / color management) — see P2_RESULTS.md.
//
// Layout (all flattened device arrays):
//   edges[ei]  = float4(x0,y0,x1,y1)            polygon boundary segments
//   poly: edge_start[pi], edge_count[pi], color_bgr[pi] (packed 0xBBGGRR), page_id[pi]
//   pages share a fixed (W,H); output buffer is N*H*W*3 uint8 (BGR), painter order = pi ascending
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) \
  throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e)); } while(0)

// Crossing-number test: is (px,py) inside the polygon's edge set? (even-odd rule)
__device__ __forceinline__ bool point_in(const float4* edges, int es, int ec, float px, float py) {
  bool inside = false;
  for (int i = 0; i < ec; ++i) {
    float ax = edges[es+i].x, ay = edges[es+i].y, bx = edges[es+i].z, by = edges[es+i].w;
    bool cond = ((ay > py) != (by > py));
    if (cond) {
      float xint = ax + (py - ay) / (by - ay) * (bx - ax);
      if (px < xint) inside = !inside;
    }
  }
  return inside;
}

// One thread per output pixel of one page. Supersample SSxSS for AA, painter-order src-over.
__global__ void raster_fill_kernel(
    uint8_t* out, int W, int H, int SS,
    const float4* edges, const float4* bbox,
    const int* edge_start, const int* edge_count, const uint32_t* color, const int* poly_page,
    const int* page_poly_start, const int* page_poly_count, int n_pages) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  int pg = blockIdx.z;
  if (px >= W || py >= H || pg >= n_pages) return;

  float accB = 255.f, accG = 255.f, accR = 255.f;     // white background
  int ps = page_poly_start[pg], pc = page_poly_count[pg];
  float inv = 1.0f / (SS * SS);
  float fpx = px + 0.5f, fpy = py + 0.5f;
  for (int k = 0; k < pc; ++k) {                       // painter's order
    int pi = ps + k;
    // Bounding-box early-out: skip polygons that cannot cover this pixel.
    float4 bb = bbox[pi];
    if (fpx < bb.x - 1 || fpx > bb.z + 1 || fpy < bb.y - 1 || fpy > bb.w + 1) continue;
    int es = edge_start[pi], ec = edge_count[pi];
    int hits = 0;
    for (int sy = 0; sy < SS; ++sy)
      for (int sx = 0; sx < SS; ++sx) {
        float fx = px + (sx + 0.5f) / SS, fy = py + (sy + 0.5f) / SS;
        if (point_in(edges, es, ec, fx, fy)) ++hits;
      }
    if (!hits) continue;
    float cov = hits * inv;                            // coverage in [0,1] = alpha
    uint32_t c = color[pi];
    float cb = (c & 0xFF), cg = ((c >> 8) & 0xFF), cr = ((c >> 16) & 0xFF);
    accB = cb * cov + accB * (1 - cov);
    accG = cg * cov + accG * (1 - cov);
    accR = cr * cov + accR * (1 - cov);
  }
  size_t off = ((size_t)pg * H + py) * W * 3 + (size_t)px * 3;
  out[off + 0] = (uint8_t)(accB + 0.5f);
  out[off + 1] = (uint8_t)(accG + 0.5f);
  out[off + 2] = (uint8_t)(accR + 0.5f);
}

struct Scene {            // host-side flattened scene for a batch of equal-size pages
  int W, H, n_pages;
  std::vector<float> edges;          // 4 per edge
  std::vector<float> bbox;           // 4 per polygon (minx,miny,maxx,maxy)
  std::vector<int> edge_start, edge_count, poly_page;
  std::vector<uint32_t> color;
  std::vector<int> page_poly_start, page_poly_count;
};

struct Timings { double h2d=0, kernel=0, d2h=0; };

// Render a batch on device `dev`; returns host BGR buffer (N*H*W*3). Optionally time kernel only.
static std::vector<uint8_t> render_on_device(const Scene& s, int SS, int dev, double* ms_kernel) {
  CUDA_OK(cudaSetDevice(dev));
  float4 *d_edges, *d_bbox; int *d_es,*d_ec,*d_pp,*d_pps,*d_ppc; uint32_t* d_col; uint8_t* d_out;
  size_t n_edges = s.edges.size()/4, n_poly = s.edge_start.size();
  size_t out_sz = (size_t)s.n_pages * s.H * s.W * 3;
  CUDA_OK(cudaMalloc(&d_edges, n_edges*sizeof(float4)));
  CUDA_OK(cudaMalloc(&d_bbox, n_poly*sizeof(float4)));
  CUDA_OK(cudaMalloc(&d_es, n_poly*sizeof(int)));
  CUDA_OK(cudaMalloc(&d_ec, n_poly*sizeof(int)));
  CUDA_OK(cudaMalloc(&d_pp, n_poly*sizeof(int)));
  CUDA_OK(cudaMalloc(&d_col, n_poly*sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&d_pps, s.n_pages*sizeof(int)));
  CUDA_OK(cudaMalloc(&d_ppc, s.n_pages*sizeof(int)));
  CUDA_OK(cudaMalloc(&d_out, out_sz));
  CUDA_OK(cudaMemcpy(d_edges, s.edges.data(), n_edges*sizeof(float4), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_bbox, s.bbox.data(), n_poly*sizeof(float4), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_es, s.edge_start.data(), n_poly*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_ec, s.edge_count.data(), n_poly*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_pp, s.poly_page.data(), n_poly*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_col, s.color.data(), n_poly*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_pps, s.page_poly_start.data(), s.n_pages*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_ppc, s.page_poly_count.data(), s.n_pages*sizeof(int), cudaMemcpyHostToDevice));

  dim3 blk(16, 16, 1);
  dim3 grd((s.W+15)/16, (s.H+15)/16, s.n_pages);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  raster_fill_kernel<<<grd, blk>>>(d_out, s.W, s.H, SS, d_edges, d_bbox, d_es, d_ec, d_col, d_pp,
                                   d_pps, d_ppc, s.n_pages);
  cudaEventRecord(e1);
  CUDA_OK(cudaDeviceSynchronize());
  float kms = 0; cudaEventElapsedTime(&kms, e0, e1); if (ms_kernel) *ms_kernel = kms;

  std::vector<uint8_t> out(out_sz);
  CUDA_OK(cudaMemcpy(out.data(), d_out, out_sz, cudaMemcpyDeviceToHost));
  cudaFree(d_edges); cudaFree(d_bbox); cudaFree(d_es); cudaFree(d_ec); cudaFree(d_pp);
  cudaFree(d_col); cudaFree(d_pps); cudaFree(d_ppc); cudaFree(d_out);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return out;
}

// Like render_on_device but instruments H2D/kernel/D2H separately and can skip the D2H copy
// (modeling P3's zero-copy hand-off where rasters stay resident for the downstream GPU model).
static Timings time_render_on_device(const Scene& s, int SS, int dev, bool copyback) {
  CUDA_OK(cudaSetDevice(dev));
  float4 *d_edges, *d_bbox; int *d_es,*d_ec,*d_pp,*d_pps,*d_ppc; uint32_t* d_col; uint8_t* d_out;
  size_t n_edges = s.edges.size()/4, n_poly = s.edge_start.size();
  size_t out_sz = (size_t)s.n_pages * s.H * s.W * 3;
  CUDA_OK(cudaMalloc(&d_edges, n_edges*sizeof(float4)));
  CUDA_OK(cudaMalloc(&d_bbox, n_poly*sizeof(float4)));
  CUDA_OK(cudaMalloc(&d_es, n_poly*sizeof(int))); CUDA_OK(cudaMalloc(&d_ec, n_poly*sizeof(int)));
  CUDA_OK(cudaMalloc(&d_pp, n_poly*sizeof(int))); CUDA_OK(cudaMalloc(&d_col, n_poly*sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&d_pps, s.n_pages*sizeof(int))); CUDA_OK(cudaMalloc(&d_ppc, s.n_pages*sizeof(int)));
  CUDA_OK(cudaMalloc(&d_out, out_sz));
  cudaEvent_t a,b,c,d; cudaEventCreate(&a);cudaEventCreate(&b);cudaEventCreate(&c);cudaEventCreate(&d);
  cudaEventRecord(a);
  cudaMemcpy(d_edges,s.edges.data(),n_edges*sizeof(float4),cudaMemcpyHostToDevice);
  cudaMemcpy(d_bbox,s.bbox.data(),n_poly*sizeof(float4),cudaMemcpyHostToDevice);
  cudaMemcpy(d_es,s.edge_start.data(),n_poly*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ec,s.edge_count.data(),n_poly*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_pp,s.poly_page.data(),n_poly*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_col,s.color.data(),n_poly*sizeof(uint32_t),cudaMemcpyHostToDevice);
  cudaMemcpy(d_pps,s.page_poly_start.data(),s.n_pages*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ppc,s.page_poly_count.data(),s.n_pages*sizeof(int),cudaMemcpyHostToDevice);
  cudaEventRecord(b);
  dim3 blk(16,16,1), grd((s.W+15)/16,(s.H+15)/16,s.n_pages);
  raster_fill_kernel<<<grd,blk>>>(d_out,s.W,s.H,SS,d_edges,d_bbox,d_es,d_ec,d_col,d_pp,d_pps,d_ppc,s.n_pages);
  cudaEventRecord(c);
  std::vector<uint8_t> host;
  if (copyback) { host.resize(out_sz); cudaMemcpy(host.data(),d_out,out_sz,cudaMemcpyDeviceToHost); }
  cudaEventRecord(d);
  CUDA_OK(cudaDeviceSynchronize());
  Timings t;
  float h2d=0,ker=0,d2h=0;
  cudaEventElapsedTime(&h2d,a,b); cudaEventElapsedTime(&ker,b,c); cudaEventElapsedTime(&d2h,c,d);
  t.h2d=h2d; t.kernel=ker; t.d2h=d2h;
  cudaFree(d_edges);cudaFree(d_bbox);cudaFree(d_es);cudaFree(d_ec);cudaFree(d_pp);cudaFree(d_col);
  cudaFree(d_pps);cudaFree(d_ppc);cudaFree(d_out);
  cudaEventDestroy(a);cudaEventDestroy(b);cudaEventDestroy(c);cudaEventDestroy(d);
  return t;
}

// Build a Scene from flat python arrays.
static Scene make_scene(int W, int H, int n_pages,
                        const std::vector<float>& edges, const std::vector<int>& edge_start,
                        const std::vector<int>& edge_count, const std::vector<uint32_t>& color,
                        const std::vector<int>& page_poly_start,
                        const std::vector<int>& page_poly_count,
                        const std::vector<int>& poly_page) {
  Scene s; s.W=W; s.H=H; s.n_pages=n_pages; s.edges=edges; s.edge_start=edge_start;
  s.edge_count=edge_count; s.color=color; s.page_poly_start=page_poly_start;
  s.page_poly_count=page_poly_count; s.poly_page=poly_page;
  // Precompute per-polygon bounding boxes from edge endpoints (for the kernel early-out).
  size_t n_poly = edge_start.size();
  s.bbox.resize(n_poly * 4);
  for (size_t pi = 0; pi < n_poly; ++pi) {
    float mnx=1e30f, mny=1e30f, mxx=-1e30f, mxy=-1e30f;
    int es = edge_start[pi], ec = edge_count[pi];
    for (int i = 0; i < ec; ++i) {
      float ax=edges[(es+i)*4], ay=edges[(es+i)*4+1], bx=edges[(es+i)*4+2], by=edges[(es+i)*4+3];
      mnx=fminf(mnx,fminf(ax,bx)); mny=fminf(mny,fminf(ay,by));
      mxx=fmaxf(mxx,fmaxf(ax,bx)); mxy=fmaxf(mxy,fmaxf(ay,by));
    }
    s.bbox[pi*4]=mnx; s.bbox[pi*4+1]=mny; s.bbox[pi*4+2]=mxx; s.bbox[pi*4+3]=mxy;
  }
  return s;
}

static nb::object to_ndarray(std::vector<uint8_t>&& buf, int n, int H, int W) {
  auto* keep = new std::vector<uint8_t>(std::move(buf));
  nb::capsule owner(keep, [](void* p) noexcept { delete static_cast<std::vector<uint8_t>*>(p); });
  size_t shape[4] = {(size_t)n,(size_t)H,(size_t)W,3};
  return nb::cast(nb::ndarray<nb::numpy, uint8_t>(keep->data(), 4, shape, owner));
}

// ---- P3' DLPack device handoff -------------------------------------------------------------
// Reduction kernel: per-image sum of bytes (proves we read device memory directly).
__global__ void sum_bytes_kernel(const uint8_t* data, size_t n, unsigned long long* out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, (unsigned long long)data[i]);
}

// Owns a CUDA allocation; frees on destruction (shared so __dlpack__ exports can outlive it).
struct DeviceBuf {
  uint8_t* d = nullptr; int dev = 0;
  ~DeviceBuf() { if (d) { cudaSetDevice(dev); cudaFree(d); } }
};

// Python-visible device tensor handle: torch-consumable via __dlpack__, and accepted by our
// own GPU ops directly. This is the payload that travels between fused operators (no host copy).
struct DeviceImage {
  std::shared_ptr<DeviceBuf> buf;
  std::vector<size_t> shape;
  size_t n = 0;
  int dev = 0;
  uintptr_t data_ptr() const { return reinterpret_cast<uintptr_t>(buf->d); }
  std::vector<size_t> get_shape() const { return shape; }
  std::string device() const { return "cuda:" + std::to_string(dev); }
  // DLPack export: torch.from_dlpack(device_image) wraps the SAME memory zero-copy.
  nb::ndarray<nb::device::cuda, uint8_t> dlpack(nb::kwargs) {
    auto* keep = new std::shared_ptr<DeviceBuf>(buf);
    nb::capsule owner(keep, [](void* p) noexcept {
      delete static_cast<std::shared_ptr<DeviceBuf>*>(p);
    });
    return nb::ndarray<nb::device::cuda, uint8_t>(buf->d, shape.size(), shape.data(), owner,
                                                  nullptr, nb::dtype<uint8_t>(),
                                                  nb::device::cuda::value, dev);
  }
  std::tuple<int, int> dlpack_device() const { return {2 /*kDLCUDA*/, dev}; }
};

NB_MODULE(_gpu_raster, m) {
  m.doc() = "CUDA batched AA polygon-fill rasterizer (P2) + DLPack device handoff (P3').";

  m.def("device_count", []{ int n=0; cudaGetDeviceCount(&n); return n; });

  nb::class_<DeviceImage>(m, "DeviceImage")
      .def_prop_ro("data_ptr", &DeviceImage::data_ptr)
      .def_prop_ro("shape", &DeviceImage::get_shape)
      .def_prop_ro("device", &DeviceImage::device)
      .def("__dlpack__", &DeviceImage::dlpack)
      .def("__dlpack_device__", &DeviceImage::dlpack_device);

  // PRODUCER: copy a host array to the GPU once, return a DeviceImage. This is the single
  // unavoidable H2D when rasterization is on CPU (PDFium); everything downstream stays on-device.
  m.def("upload_to_device", [](nb::ndarray<nb::c_contig, const uint8_t> host, int dev) {
    CUDA_OK(cudaSetDevice(dev));
    size_t n = host.size();
    auto bufp = std::make_shared<DeviceBuf>(); bufp->dev = dev;
    CUDA_OK(cudaMalloc(&bufp->d, n));
    CUDA_OK(cudaMemcpy(bufp->d, host.data(), n, cudaMemcpyHostToDevice));
    DeviceImage img; img.buf = bufp; img.n = n; img.dev = dev;
    img.shape.resize(host.ndim());
    for (size_t i = 0; i < host.ndim(); ++i) img.shape[i] = host.shape(i);
    return img;
  }, "host"_a, "dev"_a = 0);

  // CONSUMER: read the SAME device buffer (zero-copy), run a kernel, return (device_ptr, mean).
  // Models a downstream GPU op (YOLOX/OCR) consuming the rasterizer output with no host round-trip.
  m.def("consume_mean", [](DeviceImage& a) {
    CUDA_OK(cudaSetDevice(a.dev));
    unsigned long long* d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_out, sizeof(unsigned long long)));
    CUDA_OK(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    int threads = 256, blocks = (int)((a.n + threads - 1) / threads);
    sum_bytes_kernel<<<blocks, threads>>>(a.buf->d, a.n, d_out);
    CUDA_OK(cudaDeviceSynchronize());
    unsigned long long h_out = 0;
    CUDA_OK(cudaMemcpy(&h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    nb::dict r;
    r["device_ptr"] = a.data_ptr();
    r["n"] = a.n;
    r["mean"] = a.n ? (double)h_out / (double)a.n : 0.0;
    return r;
  }, "a"_a);

  // render(...) -> (ndarray[n,H,W,3], kernel_ms). Single device.
  m.def("render", [](int W, int H, int n_pages, int SS, int dev,
                     std::vector<float> edges, std::vector<int> edge_start,
                     std::vector<int> edge_count, std::vector<uint32_t> color,
                     std::vector<int> page_poly_start, std::vector<int> page_poly_count,
                     std::vector<int> poly_page) {
    Scene s = make_scene(W,H,n_pages,edges,edge_start,edge_count,color,
                         page_poly_start,page_poly_count,poly_page);
    double kms=0; auto out = render_on_device(s, SS, dev, &kms);
    return nb::make_tuple(to_ndarray(std::move(out), n_pages, H, W), kms);
  }, "W"_a,"H"_a,"n_pages"_a,"SS"_a,"dev"_a,"edges"_a,"edge_start"_a,"edge_count"_a,
     "color"_a,"page_poly_start"_a,"page_poly_count"_a,"poly_page"_a);

  // time_render(...) -> dict of (h2d_ms, kernel_ms, d2h_ms). copyback=False models zero-copy.
  m.def("time_render", [](int W,int H,int n_pages,int SS,int dev,bool copyback,
                          std::vector<float> edges, std::vector<int> edge_start,
                          std::vector<int> edge_count, std::vector<uint32_t> color,
                          std::vector<int> page_poly_start, std::vector<int> page_poly_count,
                          std::vector<int> poly_page) {
    Scene s = make_scene(W,H,n_pages,edges,edge_start,edge_count,color,
                         page_poly_start,page_poly_count,poly_page);
    Timings t;
    { nb::gil_scoped_release rel; t = time_render_on_device(s, SS, dev, copyback); }
    nb::dict d; d["h2d_ms"]=t.h2d; d["kernel_ms"]=t.kernel; d["d2h_ms"]=t.d2h; return d;
  }, "W"_a,"H"_a,"n_pages"_a,"SS"_a,"dev"_a,"copyback"_a,"edges"_a,"edge_start"_a,"edge_count"_a,
     "color"_a,"page_poly_start"_a,"page_poly_count"_a,"poly_page"_a);
}
