"""Patch Creative Relight.exe: replace module code objects inside the embedded
PyInstaller PYZ archive with newly compiled versions.

The PYZ is the LAST data entry in the CArchive, so all other entries keep their
positions byte-for-byte; only the PYZ payload, the TOC size fields and the
cookie offsets are rewritten.
"""
import struct, zlib, marshal, sys, os, shutil

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXE_IN = "repo/Creative Relight.exe"
EXE_OUT = "dist/Creative Relight.exe"

# module name -> source path (compiled with the running Python: 3.13)
PATCH_MODULES = {
    "src.utils.image_utils":        "repo/src/utils/image_utils.py",
    "src.utils.file_handler":       "repo/src/utils/file_handler.py",
    "src.processors.image_processor": "repo/src/processors/image_processor.py",
    "src.processors.sequence_processor": "repo/src/processors/sequence_processor.py",
    "ui.main_window":               "repo/ui/main_window.py",
    "ui.media_preview":             "repo/ui/media_preview.py",
}

MAGIC = b"MEI\x0c\x0b\x0a\x0b\x0e"
CA_COOKIE = "!8sIIII64s"

os.makedirs("dist", exist_ok=True)
with open(EXE_IN, "rb") as f:
    exe = f.read()

idx = exe.rfind(MAGIC)
magic, pkglen, toc_off, toclen, pyvers, pylib = struct.unpack(CA_COOKIE, exe[idx: idx + 88])
archive_start = idx + 88 - pkglen
print(f"cookie@{idx} archive_start={archive_start} toc_off={toc_off} toclen={toclen}")

ctoc = exe[archive_start + toc_off: archive_start + toc_off + toclen]

# ---- parse CArchive TOC ----
entries = []  # (name, type, entry_pos, csize, usize, cflag, raw_entry_tail)
pos = 0
pyz_idx = None
while pos < len(ctoc):
    (entry_len,) = struct.unpack("!i", ctoc[pos:pos + 4])
    entry_pos, csize, usize, cflag, typ = struct.unpack("!IIIBc", ctoc[pos + 4:pos + 18])
    name = ctoc[pos + 18: pos + entry_len].rstrip(b"\x00").decode("utf-8", "replace")
    entries.append(dict(name=name, typ=typ.decode(), pos=entry_pos, csize=csize,
                        usize=usize, flag=cflag, raw=ctoc[pos:pos + entry_len]))
    if typ.decode() in ("z", "Z"):
        pyz_idx = len(entries) - 1
    pos += entry_len

pyz_e = entries[pyz_idx]
print(f"PYZ entry: pos={pyz_e['pos']} csize={pyz_e['csize']} flag={pyz_e['flag']}")
assert pyz_e["pos"] + pyz_e["csize"] == toc_off, "PYZ is not the last data entry!"

pyz = exe[archive_start + pyz_e["pos"]: archive_start + pyz_e["pos"] + pyz_e["csize"]]
assert pyz[:4] == b"PYZ\x00"
pyc_magic = pyz[4:8]
(old_tocpos,) = struct.unpack("!I", pyz[8:12])
print(f"PYZ pyc magic: {pyc_magic.hex()}")

# ---- parse PYZ TOC ----
pyz_toc = marshal.loads(pyz[old_tocpos:])
pyz_toc_map = dict(pyz_toc)
print(f"PYZ modules: {len(pyz_toc)}")

# ---- compile replacement modules ----
new_blobs = {}
for mod, src_path in PATCH_MODULES.items():
    assert mod in pyz_toc_map, f"{mod} not in PYZ"
    with open(src_path, "rb") as f:
        src = f.read()
    # keep original co_filename for nicer tracebacks
    orig = marshal.loads(zlib.decompress(
        pyz[pyz_toc_map[mod][1]: pyz_toc_map[mod][1] + pyz_toc_map[mod][2]]))
    co = compile(src, orig.co_filename, "exec")
    blob = zlib.compress(marshal.dumps(co), 9)
    new_blobs[mod] = blob
    print(f"compiled {mod}: {len(blob)} bytes (was {pyz_toc_map[mod][2]})")

# ---- rebuild PYZ body: keep module data in TOC order, replace targets ----
order = sorted(range(len(pyz_toc)), key=lambda i: pyz_toc[i][1][1])
body = bytearray()
new_toc = []
for i in order:
    name, (typ, p, l) = pyz_toc[i]
    if name in new_blobs:
        data = new_blobs[name]
    else:
        data = pyz[p: p + l]
    new_pos = 12 + len(body)  # data area starts after 12-byte header (magic+pycmagic+tocpos)
    body += data
    new_toc.append((name, (typ, new_pos, len(data))))
new_toc_bytes = marshal.dumps(new_toc)
new_pyz = b"PYZ\x00" + pyc_magic + struct.pack("!I", 12 + len(body)) + bytes(body) + new_toc_bytes
print(f"new PYZ: {len(new_pyz)} bytes (was {len(pyz)})")

# sanity: round-trip every patched module from the new PYZ
nt = dict(new_toc)
for mod in PATCH_MODULES:
    t, p, l = nt[mod]
    co = marshal.loads(zlib.decompress(new_pyz[p:p + l]))
    assert co.co_name == "<module>"
print("round-trip check OK")

# ---- rebuild CArchive: all entries before PYZ are untouched ----
pyz_data_off = archive_start + pyz_e["pos"]
out = bytearray(exe[:pyz_data_off])          # everything up to (not incl.) old PYZ
out += new_pyz                               # new PYZ payload
new_toc_off = pyz_data_off - archive_start + len(new_pyz)

# rebuild TOC bytes: same entries; patch PYZ csize/usize
toc_out = bytearray()
for e in entries:
    if e is pyz_e:
        csize = usize = len(new_pyz)
        head = struct.pack("!IIIBc", e["pos"], csize, usize, e["flag"], e["typ"].encode())
        # entry layout: [0:4]=entry_len, [4:18]=fields, [18:]=name(+pad)
        entry = e["raw"][:4] + head + e["raw"][18:]
        assert len(entry) == len(e["raw"]), "TOC entry length changed unexpectedly"
    else:
        entry = e["raw"]
    toc_out += entry
assert len(toc_out) == toclen, "TOC length changed"
out += toc_out

# cookie: same fields except pkglen and toc_off
new_pkglen = len(out) + 88 - archive_start
cookie = struct.pack(CA_COOKIE, MAGIC, new_pkglen, new_toc_off, toclen, pyvers, pylib)
out += cookie

with open(EXE_OUT, "wb") as f:
    f.write(out)
print(f"\nwrote {EXE_OUT}: {len(out)/1e6:.1f} MB (was {len(exe)/1e6:.1f} MB)")

# ---- final verification: reparse the output exe ----
with open(EXE_OUT, "rb") as f:
    exe2 = f.read()
idx2 = exe2.rfind(MAGIC)
m2, pkglen2, toc_off2, toclen2, pyvers2, pylib2 = struct.unpack(CA_COOKIE, exe2[idx2: idx2 + 88])
assert m2 == MAGIC and pkglen2 == len(exe2) - (idx2 + 88 - pkglen2), "cookie mismatch"
arch2 = idx2 + 88 - pkglen2
ctoc2 = exe2[arch2 + toc_off2: arch2 + toc_off2 + toclen2]
p2 = 0
pyz2 = None
while p2 < len(ctoc2):
    (el,) = struct.unpack("!i", ctoc2[p2:p2 + 4])
    ep, cs, us, cf, ty = struct.unpack("!IIIBc", ctoc2[p2 + 4:p2 + 18])
    nm = ctoc2[p2 + 18:p2 + el].rstrip(b"\x00").decode()
    if ty.decode() in ("z", "Z"):
        raw2 = exe2[arch2 + ep: arch2 + ep + cs]
        (tp,) = struct.unpack("!I", raw2[8:12])
        tl = marshal.loads(raw2[tp:])
        print(f"verify: PYZ '{nm}' OK, {len(tl)} modules")
        d = dict(tl)
        for mod in PATCH_MODULES:
            t, pp, ll = d[mod]
            co = marshal.loads(zlib.decompress(raw2[pp:pp + ll]))
            print(f"verify: {mod} -> {co.co_filename} OK")
    p2 += el
print("PATCH COMPLETE")
