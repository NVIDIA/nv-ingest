// P1 — Native PDFium bindings mirroring the pypdfium2 surface the codebase uses.
// Classes: PdfDocument, PdfPage, PdfTextPage, PdfBitmap, PdfPageObject.
// Read path (open/pages/render/text/objects/image/metadata) + write path (new/import/save).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fpdfview.h"
#include "fpdf_edit.h"
#include "fpdf_text.h"
#include "fpdf_doc.h"
#include "fpdf_ppo.h"
#include "fpdf_save.h"

namespace nb = nanobind;
using namespace nb::literals;

// ----- PDFium bitmap format constants (stable PDFium ABI values) --------------
static constexpr int kGray = 1, kBGR = 2, kBGRx = 3, kBGRA = 4;
static constexpr int kFPDF_ANNOT = 0x01;  // match pypdfium2 draw_annots=True default

// ----- error type -------------------------------------------------------------
struct PdfiumError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ----- one-time library init --------------------------------------------------
static void ensure_init() {
  static std::once_flag once;
  std::call_once(once, [] {
    FPDF_LIBRARY_CONFIG cfg{};
    cfg.version = 2;
    cfg.m_pUserFontPaths = nullptr;
    cfg.m_pIsolate = nullptr;
    cfg.m_v8EmbedderSlot = 0;
    FPDF_InitLibraryWithConfig(&cfg);
  });
}

static int channels_for(int fmt) {
  switch (fmt) { case kGray: return 1; case kBGR: return 3; default: return 4; }
}
static const char* mode_for(int fmt) {
  switch (fmt) { case kGray: return "L"; case kBGR: return "BGR";
                 case kBGRx: return "BGRX"; default: return "BGRA"; }
}

static std::string utf16le_to_utf8(const std::vector<uint16_t>& u16) {
  std::string out;
  out.reserve(u16.size());
  for (size_t i = 0; i < u16.size(); ++i) {
    uint32_t cp = u16[i];
    if (cp == 0) break;  // NUL terminator
    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < u16.size()) {
      uint32_t lo = u16[i + 1];
      if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); ++i; }
    }
    if (cp < 0x80) out.push_back((char)cp);
    else if (cp < 0x800) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
    else { out.push_back((char)(0xF0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
  }
  return out;
}

// =============================================================================
// PdfBitmap — owns a contiguous (H,W,C) uint8 buffer.
// =============================================================================
struct PdfBitmap {
  std::shared_ptr<std::vector<uint8_t>> data;
  int w = 0, h = 0, c = 0;
  std::string mode;

  // Build from an FPDF_BITMAP (copies rows out, dropping any stride padding).
  static PdfBitmap from_fpdf(FPDF_BITMAP bmp) {
    PdfBitmap o;
    int fmt = FPDFBitmap_GetFormat(bmp);
    o.w = FPDFBitmap_GetWidth(bmp);
    o.h = FPDFBitmap_GetHeight(bmp);
    o.c = channels_for(fmt);
    o.mode = mode_for(fmt);
    int stride = FPDFBitmap_GetStride(bmp);
    const uint8_t* src = static_cast<const uint8_t*>(FPDFBitmap_GetBuffer(bmp));
    o.data = std::make_shared<std::vector<uint8_t>>((size_t)o.w * o.h * o.c);
    int rowbytes = o.w * o.c;
    for (int y = 0; y < o.h; ++y)
      std::memcpy(o.data->data() + (size_t)y * rowbytes, src + (size_t)y * stride, rowbytes);
    return o;
  }

  nb::object to_numpy() {
    auto* keep = new std::shared_ptr<std::vector<uint8_t>>(data);
    nb::capsule owner(keep, [](void* p) noexcept {
      delete static_cast<std::shared_ptr<std::vector<uint8_t>>*>(p);
    });
    if (c == 1) {
      size_t shape[2] = {(size_t)h, (size_t)w};
      return nb::cast(nb::ndarray<nb::numpy, uint8_t>(data->data(), 2, shape, owner));
    }
    size_t shape[3] = {(size_t)h, (size_t)w, (size_t)c};
    return nb::cast(nb::ndarray<nb::numpy, uint8_t>(data->data(), 3, shape, owner));
  }
};

// =============================================================================
// PdfTextPage
// =============================================================================
struct PdfTextPage {
  FPDF_TEXTPAGE tp = nullptr;
  FPDF_PAGE page = nullptr;
  // Match pypdfium2.get_text_bounded: bounded text over the page bbox (media∩crop), not raw
  // char-index order (which can differ for multi-column / reordered content).
  std::string get_text_bounded() {
    if (!tp || !page) return "";
    FS_RECTF r{};
    if (!FPDF_GetPageBoundingBox(page, &r)) return "";
    int n = FPDFText_GetBoundedText(tp, r.left, r.top, r.right, r.bottom, nullptr, 0);
    if (n <= 0) return "";
    std::vector<uint16_t> buf(n, 0);
    FPDFText_GetBoundedText(tp, r.left, r.top, r.right, r.bottom, buf.data(), n);
    return utf16le_to_utf8(buf);
  }
  void close() { if (tp) { FPDFText_ClosePage(tp); tp = nullptr; } }
};

// =============================================================================
// PdfPageObject
// =============================================================================
struct PdfPageObject {
  FPDF_PAGEOBJECT obj = nullptr;
  FPDF_DOCUMENT doc = nullptr;
  FPDF_PAGE page = nullptr;
  nb::object keepalive;  // hold the owning PdfPage so handles stay valid

  int type() const { return FPDFPageObj_GetType(obj); }

  std::tuple<float, float, float, float> get_pos() const {
    float l = 0, b = 0, r = 0, t = 0;
    FPDFPageObj_GetBounds(obj, &l, &b, &r, &t);
    return {l, b, r, t};  // (left, bottom, right, top) — matches pypdfium2
  }

  std::tuple<int, int> get_size() const {
    unsigned int w = 0, hh = 0;
    if (FPDFImageObj_GetImagePixelSize(obj, &w, &hh)) return {(int)w, (int)hh};
    float l = 0, b = 0, r = 0, t = 0;
    FPDFPageObj_GetBounds(obj, &l, &b, &r, &t);
    return {(int)(r - l), (int)(t - b)};
  }

  nb::object get_bitmap(bool render) {
    FPDF_BITMAP bmp = render ? FPDFImageObj_GetRenderedBitmap(doc, page, obj)
                             : FPDFImageObj_GetBitmap(obj);
    if (!bmp) return nb::none();
    PdfBitmap o = PdfBitmap::from_fpdf(bmp);
    FPDFBitmap_Destroy(bmp);
    return nb::cast(std::move(o));
  }
};

// =============================================================================
// PdfPage
// =============================================================================
struct PdfPage {
  FPDF_PAGE page = nullptr;
  FPDF_DOCUMENT doc = nullptr;
  nb::object keepalive;  // owning PdfDocument

  double get_width() const { return FPDF_GetPageWidthF(page); }
  double get_height() const { return FPDF_GetPageHeightF(page); }
  std::tuple<double, double> get_size() const { return {get_width(), get_height()}; }
  int get_rotation() const { return FPDFPage_GetRotation(page) * 90; }

  PdfBitmap render(double scale, int rotation) {
    int w = (int)std::ceil(get_width() * scale);
    int h = (int)std::ceil(get_height() * scale);
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    // (rotation degrees / 90) when applying at draw time PDFium swaps W/H for 90/270.
    int rot = ((rotation / 90) % 4 + 4) % 4;
    int bw = (rot % 2) ? h : w, bh = (rot % 2) ? w : h;
    FPDF_BITMAP bmp = FPDFBitmap_CreateEx(bw, bh, kBGR, nullptr, 0);
    if (!bmp) throw PdfiumError("FPDFBitmap_CreateEx failed");
    FPDFBitmap_FillRect(bmp, 0, 0, bw, bh, 0xFFFFFFFF);
    FPDF_RenderPageBitmap(bmp, page, 0, 0, bw, bh, rot, kFPDF_ANNOT);
    PdfBitmap o = PdfBitmap::from_fpdf(bmp);
    FPDFBitmap_Destroy(bmp);
    return o;
  }

  PdfTextPage get_textpage() {
    PdfTextPage t;
    t.tp = FPDFText_LoadPage(page);
    if (!t.tp) throw PdfiumError("FPDFText_LoadPage failed");
    t.page = page;
    return t;
  }

  // get_objects(filter=tuple|None, max_depth=0)
  std::vector<PdfPageObject> get_objects(nb::object filter, int max_depth);

  void close() { if (page) { FPDF_ClosePage(page); page = nullptr; } }
};

static bool type_allowed(int t, const std::vector<int>& filt) {
  if (filt.empty()) return true;
  for (int f : filt) if (f == t) return true;
  return false;
}

static void collect_objects(FPDF_PAGE page, FPDF_DOCUMENT doc, FPDF_PAGEOBJECT container,
                            bool is_form, const std::vector<int>& filt, int depth, int max_depth,
                            nb::object keepalive, std::vector<PdfPageObject>& out) {
  int count = is_form ? (int)FPDFFormObj_CountObjects(container) : FPDFPage_CountObjects(page);
  for (int i = 0; i < count; ++i) {
    FPDF_PAGEOBJECT o = is_form ? FPDFFormObj_GetObject(container, (unsigned long)i)
                                : FPDFPage_GetObject(page, i);
    if (!o) continue;
    int t = FPDFPageObj_GetType(o);
    if (type_allowed(t, filt)) {
      PdfPageObject po; po.obj = o; po.doc = doc; po.page = page; po.keepalive = keepalive;
      out.push_back(std::move(po));
    }
    // FPDF_PAGEOBJ_FORM == 5; recurse if within depth budget.
    if (t == 5 && (depth + 1) < max_depth)
      collect_objects(page, doc, o, true, filt, depth + 1, max_depth, keepalive, out);
  }
}

std::vector<PdfPageObject> PdfPage::get_objects(nb::object filter, int max_depth) {
  std::vector<int> filt;
  if (!filter.is_none())
    for (auto h : filter) filt.push_back(nb::cast<int>(nb::borrow(h)));
  if (max_depth < 1) max_depth = 1;
  std::vector<PdfPageObject> out;
  collect_objects(page, doc, nullptr, false, filt, 0, max_depth, keepalive, out);
  return out;
}

// =============================================================================
// PdfDocument
// =============================================================================
struct FileWriter { FPDF_FILEWRITE fw; std::string* buf; };
static int write_block(FPDF_FILEWRITE* pThis, const void* data, unsigned long size) {
  auto* w = reinterpret_cast<FileWriter*>(pThis);
  w->buf->append(static_cast<const char*>(data), size);
  return 1;
}

struct PdfDocument {
  FPDF_DOCUMENT doc = nullptr;
  std::shared_ptr<std::vector<uint8_t>> src;  // keep source bytes alive (PDFium does not copy)

  static std::string meta(FPDF_DOCUMENT d, const char* tag) {
    unsigned long n = FPDF_GetMetaText(d, tag, nullptr, 0);
    if (n <= 2) return "";
    std::vector<uint16_t> buf(n / 2, 0);
    FPDF_GetMetaText(d, tag, buf.data(), n);
    return utf16le_to_utf8(buf);
  }

  void open_from_object(nb::object source) {
    ensure_init();
    // Bytes-like (bytes/bytearray/memoryview/BytesIO.getbuffer) → PyBytes_FromObject copies
    // them; it raises TypeError for str/PathLike, which we treat as a filesystem path.
    PyObject* bobj = PyBytes_FromObject(source.ptr());
    if (bobj) {
      nb::bytes b = nb::steal<nb::bytes>(bobj);
      const uint8_t* p = (const uint8_t*)b.c_str();
      src = std::make_shared<std::vector<uint8_t>>(p, p + b.size());
    } else {
      PyErr_Clear();
      // BytesIO and similar: try .getvalue() before falling back to a path.
      if (nb::hasattr(source, "getvalue")) {
        nb::bytes b = nb::cast<nb::bytes>(source.attr("getvalue")());
        const uint8_t* p = (const uint8_t*)b.c_str();
        src = std::make_shared<std::vector<uint8_t>>(p, p + b.size());
      } else {
        std::string path = nb::cast<std::string>(nb::str(source));
        std::ifstream f(path, std::ios::binary);
        if (!f) throw PdfiumError("could not open file: " + path);
        src = std::make_shared<std::vector<uint8_t>>(std::istreambuf_iterator<char>(f),
                                                     std::istreambuf_iterator<char>());
      }
    }
    doc = FPDF_LoadMemDocument64(src->data(), src->size(), nullptr);
    if (!doc) throw PdfiumError("FPDF_LoadMemDocument failed (err=" + std::to_string(FPDF_GetLastError()) + ")");
  }

  static PdfDocument make_new() {
    ensure_init();
    PdfDocument d;
    d.doc = FPDF_CreateNewDocument();
    if (!d.doc) throw PdfiumError("FPDF_CreateNewDocument failed");
    return d;
  }

  int len() const { return doc ? FPDF_GetPageCount(doc) : 0; }
  int get_page_count() const { return len(); }

  PdfPage get_page(int index) {
    FPDF_PAGE p = FPDF_LoadPage(doc, index);
    if (!p) throw PdfiumError("FPDF_LoadPage failed at index " + std::to_string(index));
    PdfPage pg; pg.page = p; pg.doc = doc; pg.keepalive = nb::none();
    return pg;
  }

  nb::dict get_metadata_dict() {
    nb::dict d;
    const char* tags[] = {"Title", "Author", "Subject", "Keywords",
                          "Creator", "Producer", "CreationDate", "ModDate"};
    for (auto* t : tags) d[t] = meta(doc, t);
    return d;
  }

  void import_pages(PdfDocument& srcdoc, std::vector<int> pages) {
    std::string range;
    for (size_t i = 0; i < pages.size(); ++i) {
      if (i) range += ",";
      range += std::to_string(pages[i] + 1);  // PDFium pageranges are 1-based
    }
    int at = FPDF_GetPageCount(doc);
    FPDF_BOOL ok = FPDF_ImportPages(doc, srcdoc.doc, range.empty() ? nullptr : range.c_str(), at);
    if (!ok) throw PdfiumError("FPDF_ImportPages failed");
  }

  nb::bytes save_to_bytes() {
    std::string out;
    FileWriter w{};
    w.fw.version = 1;
    w.fw.WriteBlock = &write_block;
    w.buf = &out;
    FPDF_BOOL ok = FPDF_SaveAsCopy(doc, &w.fw, 0);
    if (!ok) throw PdfiumError("FPDF_SaveAsCopy failed");
    return nb::bytes(out.data(), out.size());
  }

  // pypdfium2-compatible: save(buf) writes to a file-like object.
  void save(nb::object buf) { buf.attr("write")(save_to_bytes()); }

  void close() {
    if (doc) { FPDF_CloseDocument(doc); doc = nullptr; }
    src.reset();
  }
};

// =============================================================================
NB_MODULE(_gpu_pdfium, m) {
  m.doc() = "Native PDFium bindings (pypdfium2-compatible drop-in core).";
  ensure_init();

  nb::exception<PdfiumError>(m, "PdfiumError");

  nb::class_<PdfBitmap>(m, "PdfBitmap")
      .def("to_numpy", &PdfBitmap::to_numpy)
      .def_ro("mode", &PdfBitmap::mode)
      .def_ro("width", &PdfBitmap::w)
      .def_ro("height", &PdfBitmap::h);

  nb::class_<PdfTextPage>(m, "PdfTextPage")
      .def("get_text_bounded", &PdfTextPage::get_text_bounded)
      .def("close", &PdfTextPage::close);

  nb::class_<PdfPageObject>(m, "PdfPageObject")
      .def_prop_ro("type", &PdfPageObject::type)
      .def("get_pos", &PdfPageObject::get_pos)
      .def("get_size", &PdfPageObject::get_size)
      .def("get_bitmap", &PdfPageObject::get_bitmap, "render"_a = true);

  nb::class_<PdfPage>(m, "PdfPage")
      .def("get_width", &PdfPage::get_width)
      .def("get_height", &PdfPage::get_height)
      .def("get_size", &PdfPage::get_size)
      .def("get_rotation", &PdfPage::get_rotation)
      .def("render", &PdfPage::render, "scale"_a = 1.0, "rotation"_a = 0)
      .def("get_textpage", &PdfPage::get_textpage, nb::keep_alive<0, 1>())
      // NB: returned objects hold raw page handles; the page must outlive them (the codebase
      // always iterates objects while the page is open). No keep_alive: the list return type
      // is not weak-referenceable.
      .def("get_objects", &PdfPage::get_objects, "filter"_a = nb::none(), "max_depth"_a = 0)
      .def("close", &PdfPage::close);

  nb::class_<PdfDocument>(m, "PdfDocument")
      .def("__init__", [](PdfDocument* self, nb::object source) {
        new (self) PdfDocument();
        self->open_from_object(source);
      }, "source"_a)
      .def_static("new", &PdfDocument::make_new)
      .def("__len__", &PdfDocument::len)
      .def("__getitem__", &PdfDocument::get_page, "index"_a, nb::keep_alive<0, 1>())
      .def("get_page", &PdfDocument::get_page, "index"_a, nb::keep_alive<0, 1>())
      .def("get_page_count", &PdfDocument::get_page_count)
      .def("get_metadata_dict", &PdfDocument::get_metadata_dict)
      .def("import_pages", &PdfDocument::import_pages, "src"_a, "pages"_a)
      .def("save", &PdfDocument::save, "buf"_a)
      .def("save_to_bytes", &PdfDocument::save_to_bytes)
      .def("close", &PdfDocument::close);
}
