import socket, urllib.request, sys
sys.stdout.reconfigure(encoding="utf-8")
try:
    ip = socket.gethostbyname("pypi.org")
    print(f"DNS: pypi.org = {ip}")
except Exception as e:
    print(f"DNS fail: {e}")
try:
    r = urllib.request.urlopen("https://pypi.org/simple/", timeout=10)
    print(f"HTTP: {r.status}")
except Exception as e:
    print(f"HTTP fail: {e}")
