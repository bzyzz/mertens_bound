"""
certio.py -- shared helpers for the certification scripts.

* Outward-safe serialisation: every certified quantity is written as an Arb ball string
  (``x.str(n, radius=True)``), which Arb guarantees to enclose the ball, and read back with
  ``arb(s)``; directed endpoints are taken after reading (``.lower()``, ``.upper()``).
  Radius-free decimals are never used as bounds.
* Schema 1 (the certificates of the initial release) stored some bounds as radius-free
  decimals. ``legacy_ball`` widens such a decimal by one unit in its last place, outward in
  both directions, which encloses any value that rounds to it.
* ``exact_int`` turns an Arb ball into an integer only if the ball contains exactly one.
* ``require`` / ``fail`` stop with a nonzero exit status; they are not ``assert`` statements,
  so they also run under ``python -O``.
* ``provenance`` / ``cache_tag`` bind caches and certificates to the parameters, the code and
  the library versions that produced them.
"""
import hashlib
import json
import os
import sys

import flint
from flint import arb

SCHEMA = 2


class CertificationError(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise CertificationError(msg)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def versions():
    return {"python": sys.version.split()[0], "python_flint": flint.__version__,
            "flint": getattr(flint, "__FLINT_VERSION__", "unknown")}


def file_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def cache_tag(params, code_paths):
    """Short tag identifying a cache namespace: parameters, schema, library versions, code."""
    blob = json.dumps({"params": params, "schema": SCHEMA, "versions": versions(),
                       "code": file_hash(*code_paths)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def provenance(params, code_paths):
    return {"schema": SCHEMA, "params": params, "versions": versions(),
            "code_sha256": {os.path.basename(p): file_hash(p) for p in code_paths}}


def ball(x, digits=30):
    """Ball string that encloses x."""
    return x.str(digits, radius=True)


def legacy_ball(s):
    """Enclosure of every real number that rounds to the radius-free decimal string s."""
    t = s.strip()
    require(not t.startswith("["), f"expected a radius-free decimal, got {s!r}")
    mant, _, exp = t.lower().partition("e")
    frac = mant.split(".")[1] if "." in mant else ""
    e = -len(frac) + (int(exp) if exp else 0)
    return arb(arb(t).mid(), arb(10) ** e + arb(t).rad())


def exact_int(x, what):
    """The unique integer in the ball x; fails if the ball contains none or several."""
    lo, hi = x.lower().ceil(), x.upper().floor()
    require(lo == hi, f"{what}: ball {x.str(20)} does not determine a unique integer")
    return int(lo.str(40, radius=False).split(".")[0])


def manifest(payloads):
    """sha256 over the sorted JSON payloads of all chunks that a summary is built from."""
    h = hashlib.sha256()
    for p in payloads:
        h.update(json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()


def cache_root(default_dir):
    """Cache root: $CERT_CACHE if set (e.g. for a run from empty caches), else the script directory."""
    return os.environ.get("CERT_CACHE", default_dir)
