"""
check_redis.py - checks that Redis (e.g. Render Key Value) is reachable and
measures how long a round trip takes.

Run from backend/:   python check_redis.py      (or from the root: npm run check:redis)

Latency matters here: every camera frame and every analysis goes through Redis,
so a slow link makes the adaptive scene react late.
"""
import os
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

url = os.environ.get("REDIS_URL", "").strip()
if not url or "paste-your" in url:
    sys.exit("REDIS_URL is not set in backend/.env. Paste Render's External URL there.")

parsed = urlparse(url)
uses_tls = parsed.scheme == "rediss"
verify = os.environ.get("REDIS_SSL_VERIFY", "1").strip().lower() not in ("0", "false", "no", "off")
# Print the host only, never the password.
print(f"Connecting to {parsed.hostname}:{parsed.port or 6379} ({'TLS' if uses_tls else 'no TLS'}) ...")

try:
    import redis
except ImportError:
    sys.exit("The 'redis' package is missing. Run: pip install -r requirements.txt")

options = {"socket_connect_timeout": 10, "socket_timeout": 10}
if uses_tls and not verify:
    options["ssl_cert_reqs"] = "none"
    print("WARNING: certificate checking is OFF (REDIS_SSL_VERIFY=0).")

try:
    client = redis.Redis.from_url(url, **options)
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        client.ping()
        timings.append((time.perf_counter() - start) * 1000)
    client.set("mirage:check", "ok", ex=30)
    assert client.get("mirage:check") == b"ok"
except Exception as exc:  # report every kind of failure the same helpful way
    print(f"\nFAILED: {type(exc).__name__}: {exc}\n")
    print("Things to check:")
    print("  1. External access is enabled on the Render instance.")
    print("  2. Your CURRENT public IP is on its allowlist (home IPs change).")
    print("  3. REDIS_URL is the EXTERNAL url and starts with rediss:// (not the internal redis:// one).")
    print("  4. If the error mentions SSL/certificate, try REDIS_SSL_VERIFY=0 once to confirm, then fix and turn it back on.")
    sys.exit(1)

median = statistics.median(timings)
print(f"OK. Round trip: median {median:.0f} ms (min {min(timings):.0f}, max {max(timings):.0f}).")
if median < 80:
    print("Latency is fine for development.")
elif median < 200:
    print("Usable, but scene changes will react a little late.")
else:
    print("Slow: scene changes will lag. A local Redis would feel much snappier.")