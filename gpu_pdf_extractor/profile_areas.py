"""Profile time spent in each capability area, per page, using the real pypdfium2 backend."""
import sys, time, io
import pypdfium2 as pdfium
import pypdfium2.raw as praw

def t(fn, n=1):
    best = None
    for _ in range(n):
        s = time.perf_counter()
        fn()
        e = time.perf_counter()
        best = (e - s) if best is None else min(best, e - s)
    return best

def profile(path, dpi=300, max_pages=40):
    raw = open(path, "rb").read()
    # 1) parse/open
    t_open = t(lambda: pdfium.PdfDocument(raw).close(), n=3)
    doc = pdfium.PdfDocument(raw)
    npages = min(len(doc), max_pages)
    scale = dpi / 72.0

    times = {"render": 0.0, "text": 0.0, "objects": 0.0, "image_decode": 0.0}
    n_imgs = 0
    for i in range(npages):
        page = doc.get_page(i)
        s = time.perf_counter(); bmp = page.render(scale=scale); arr = bmp.to_numpy(); times["render"] += time.perf_counter() - s
        s = time.perf_counter(); _ = page.get_textpage().get_text_bounded(); times["text"] += time.perf_counter() - s
        s = time.perf_counter(); objs = list(page.get_objects(filter=(praw.FPDF_PAGEOBJ_IMAGE,), max_depth=1)); times["objects"] += time.perf_counter() - s
        s = time.perf_counter()
        for o in objs:
            try:
                b = o.get_bitmap(render=True)
                if b is not None: b.to_numpy(); n_imgs += 1
            except Exception: pass
        times["image_decode"] += time.perf_counter() - s
        page.close()

    # 6) split: import 1 page into a new doc + save (per page)
    def split_one():
        single = pdfium.PdfDocument.new()
        single.import_pages(doc, pages=[0])
        buf = io.BytesIO(); single.save(buf); single.close()
    t_split = t(split_one, n=3)
    doc.close()

    print(f"\n=== {path.split('/')[-1]}  ({npages} pages profiled, {dpi} DPI) ===")
    print(f"  open/parse (whole doc): {t_open*1000:7.1f} ms  -> {t_open/npages*1000:6.2f} ms/page amortized")
    per = {k: v / npages * 1000 for k, v in times.items()}
    print(f"  render (rasterize):     {per['render']:7.2f} ms/page")
    print(f"  text extraction:        {per['text']:7.2f} ms/page")
    print(f"  object enumeration:     {per['objects']:7.2f} ms/page")
    print(f"  embedded image decode:  {per['image_decode']:7.2f} ms/page  ({n_imgs} imgs total)")
    print(f"  split (import+save):    {t_split*1000:7.2f} ms/page")
    total = per['render'] + per['text'] + per['objects'] + per['image_decode']
    print(f"  -- per-page extract subtotal (render+text+obj+img): {total:7.2f} ms")
    if total > 0:
        print(f"     render share: {per['render']/total*100:4.1f}%   text: {per['text']/total*100:4.1f}%"
              f"   obj: {per['objects']/total*100:4.1f}%   imgdec: {per['image_decode']/total*100:4.1f}%")

for p in sys.argv[1:]:
    try: profile(p)
    except Exception as e: print(f"FAILED {p}: {e}")
