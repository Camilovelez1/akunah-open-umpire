"""Minimal web view: live annotated video, score, and correction buttons.

Uses only the standard library so it runs on the Pi without extra installs,
and replaces `ssh -X`, which can't keep up with 30 fps video.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open Umpire</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 main{max-width:1000px;margin:auto;padding:12px}
 img{width:100%;border-radius:6px;background:#000}
 .score{display:flex;justify-content:space-between;font-size:2.4em;font-weight:700;margin:8px 0}
 .score small{font-size:.45em;color:#aaa;font-weight:400}
 .srv{color:#fc0}
 button{font-size:1em;padding:10px 14px;margin:4px;border-radius:6px;border:0;background:#333;color:#eee}
 button:active{background:#555}
 pre{background:#1b1b1b;padding:8px;border-radius:6px;font-size:.85em;max-height:240px;overflow:auto}
</style></head><body><main>
<img src="/stream.mjpg" alt="camera">
<div class="score"><div id="l"></div><div id="r" style="text-align:right"></div></div>
<div>
 <button onclick="cmd('point_left')">+1 left</button>
 <button onclick="cmd('point_right')">+1 right</button>
 <button onclick="cmd('undo')">Undo</button>
 <button onclick="cmd('let')">Let</button>
 <button onclick="cmd('swap')">Swap ends</button>
 <button onclick="if(confirm('Reset match?'))cmd('reset')">Reset</button>
</div>
<pre id="log"></pre>
</main><script>
let st=null;
function cmd(c){
  if(st&&c.startsWith('point_')){c='point'+(c==='point_left'?st.left_player:1-st.left_player)}
  fetch('/cmd/'+c,{method:'POST'});
}
function cell(p){
  const s=st.server===p&&!st.finished?'<span class="srv">● </span>':'';
  return s+st.names[p]+' '+st.scores[p]+' <small>games '+st.games[p]+'</small>';
}
async function poll(){
  try{st=await (await fetch('/state')).json();
    document.getElementById('l').innerHTML=cell(st.left_player);
    document.getElementById('r').innerHTML=cell(1-st.left_player);
    document.getElementById('log').textContent=(st.finished?'MATCH OVER\\n':'')+st.fps+' fps\\n'+st.log.join('\\n');
  }catch(e){}
  setTimeout(poll,300);
}
poll();
</script></body></html>
"""


class WebView:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.commands: queue.Queue[str] = queue.Queue()
        self._jpeg: bytes | None = None
        self._state: dict = {}
        self._cond = threading.Condition()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
                elif self.path == "/state":
                    with view._cond:
                        body = json.dumps(view._state).encode()
                    self._send(200, "application/json", body)
                elif self.path == "/stream.mjpg":
                    self._stream()
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self):
                if self.path.startswith("/cmd/"):
                    view.commands.put(self.path[5:])
                    self._send(204, "text/plain", b"")
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        with view._cond:
                            view._cond.wait_for(lambda: view._jpeg is not last, timeout=5)
                            jpeg = last = view._jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, jpeg: bytes, state: dict) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._state = state
            self._cond.notify_all()

    def close(self) -> None:
        self.server.shutdown()
