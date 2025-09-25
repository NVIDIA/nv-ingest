from fastapi import FastAPI, Request, Response
from typing import Any, List, Optional, Dict

app = FastAPI()

_last_upload: Optional[List[Any]] = None
_last_headers: Optional[Dict[str, Any]] = None
_fail_left: Optional[int] = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/upload")
async def upload(payload: List[Any], request: Request, response: Response):
    global _last_upload, _last_headers, _fail_left
    _last_upload = payload
    # Capture a subset of headers for testing purposes
    headers = dict(request.headers)
    _last_headers = {
        "authorization": headers.get("authorization"),
        "x-test-header": headers.get("x-test-header"),
    }
    # Allow forcing a non-200 status via header for error-path tests
    force_status = headers.get("x-force-status")
    if force_status:
        try:
            code = int(force_status)
            # Optionally set Retry-After header if provided for 408/429 tests
            retry_after = headers.get("x-retry-after")
            if retry_after is not None:
                response.headers["Retry-After"] = retry_after
            response.status_code = code
            return {"ok": False, "count": len(payload)}
        except ValueError:
            pass

    # Simulate transient failures for N requests: header x-fail-n indicates total failures to serve before success
    fail_n = headers.get("x-fail-n")
    if fail_n is not None:
        try:
            if _fail_left is None:
                _fail_left = int(fail_n)
            if _fail_left > 0:
                _fail_left -= 1
                response.status_code = 503
                return {"ok": False, "count": len(payload)}
        except ValueError:
            pass
    return {"ok": True, "count": len(payload)}


@app.get("/last")
async def last():
    return {"last": _last_upload}


@app.get("/last_headers")
async def last_headers():
    return {"headers": _last_headers}
