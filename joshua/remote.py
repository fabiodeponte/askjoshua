"""Reaching Joshua from your other devices, through Tailscale.

Tailscale puts your computers and phones on a private network of their own. Joshua keeps listening only to this
computer (127.0.0.1); when you switch access on, a small relay also listens on this computer's Tailscale address and
passes each connection on to Joshua. So only your own devices, signed in to your Tailscale, can reach it: nothing is
open to the internet or to the local Wi-Fi.
"""
import json, pathlib, shutil, socket, subprocess, sys, threading, time

PLACES = ["/Applications/Tailscale.app/Contents/MacOS/Tailscale", "/opt/homebrew/bin/tailscale", "/usr/local/bin/tailscale", "/usr/bin/tailscale",
          "C:/Program Files/Tailscale/tailscale.exe"]


def cli():
    return shutil.which("tailscale") or next((p for p in PLACES if pathlib.Path(p).exists()), None)


def info():
    """{'state': 'missing' | 'stopped' | 'running', 'ip': '100.x.y.z', 'name': 'my-mac.tail1234.ts.net'}"""
    exe = cli()
    if not exe:
        return {"state": "missing"}
    try:
        s = json.loads(subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=8).stdout or "{}")
    except Exception:
        return {"state": "stopped"}
    me = s.get("Self") or {}
    ips = [ip for ip in me.get("TailscaleIPs") or [] if "." in ip]
    if s.get("BackendState") != "Running" or not ips:
        return {"state": "stopped"}
    return {"state": "running", "ip": ips[0], "name": (me.get("DNSName") or "").rstrip(".")}


class Relay:
    """Listens on the Tailscale address and passes every connection on to Joshua on 127.0.0.1."""

    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((ip, port))
        self.sock.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                client, _ = self.sock.accept()
            except OSError:
                return  # closed
            threading.Thread(target=self._join, args=(client,), daemon=True).start()

    def _join(self, client):
        try:
            joshua = socket.create_connection(("127.0.0.1", self.port), timeout=10)
            joshua.settimeout(None)
        except OSError:
            client.close()
            return
        for a, b in ((client, joshua), (joshua, client)):
            threading.Thread(target=self._pipe, args=(a, b), daemon=True).start()

    @staticmethod
    def _pipe(src, dst):
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        self.sock.close()


_relay = {"r": None, "on": False, "port": 7860}


def _keep():
    """Every 20 seconds: if access is on, make sure the relay listens on today's Tailscale address (it can change)."""
    while True:
        now = info() if _relay["on"] else {"state": "off"}
        r = _relay["r"]
        want = now.get("ip") if now.get("state") == "running" else None
        if r and r.ip != want:
            r.close()
            _relay["r"] = r = None
        if want and not r:
            try:
                _relay["r"] = Relay(want, _relay["port"])
            except OSError:
                pass
        time.sleep(20)


threading.Thread(target=_keep, daemon=True).start()


def switch(on, port=7860):
    _relay.update(on=bool(on), port=port)
    if not on and _relay["r"]:
        _relay["r"].close()
        _relay["r"] = None
    elif on and not _relay["r"]:
        now = info()
        if now.get("state") == "running":
            try:
                _relay["r"] = Relay(now["ip"], port)
            except OSError:
                pass
    return listening()


def listening():
    return _relay["r"] is not None


def qr_svg(text):
    """A QR code of the address, to open it on a phone by pointing the camera at the screen."""
    try:
        import segno
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segno"], check=True, timeout=120)
            import segno
        except Exception:
            return ""
    import io
    out = io.BytesIO()
    segno.make(text, error="m").save(out, kind="svg", scale=4, border=2, dark="#2a2622", light=None, xmldecl=False, svgns=True)
    return out.getvalue().decode()
