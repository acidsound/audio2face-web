#!/usr/bin/env python3
"""HTTPS threaded static server with COOP/COEP for ORT Web multithreaded WASM (SIMD + SharedArrayBuffer).
Served over Tailscale-signed cert so iOS gets a secure context on the tailnet domain."""
import http.server, socketserver, os, sys, ssl

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8898
ROOT = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(ROOT, "oci2cpu16g-us-west-pheonix.tail50666.ts.net.crt")
KEY = os.path.join(ROOT, "oci2cpu16g-us-west-pheonix.tail50666.ts.net.key")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        # cross-origin isolation → enables SharedArrayBuffer (WASM multithread)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        # model binaries: download once, then serve from disk cache forever
        if self.path.endswith(".onnx"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "public, max-age=300")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    httpd = Server(("0.0.0.0", PORT), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    print(f"HTTPS serving {ROOT} on 0.0.0.0:{PORT} (threaded, COOP/COEP, TLS)")
    httpd.serve_forever()
