"""Debug PYZ structure: hexdump header, try tocpos as 4-byte and 8-byte, scan for marshal TOC."""
import struct, zlib, marshal

EXE = "repo/Creative Relight.exe"
with open(EXE, "rb") as f:
    data = f.read()

MAGIC = b"MEI\x0c\x0b\x0a\x0b\x0e"
idx = data.rfind(MAGIC)
magic, pkglen, toc_off, toclen, pyvers, pylib = struct.unpack("!8sIIII64s", data[idx: idx + 88])
archive_start = idx + 88 - pkglen
toc = data[archive_start + toc_off: archive_start + toc_off + toclen]

pos = 0
pyz_raw = None
while pos < len(toc):
    (entry_len,) = struct.unpack("!i", toc[pos:pos + 4])
    entry_pos, csize, usize, cflag, typ = struct.unpack("!IIIBc", toc[pos + 4:pos + 18])
    name = toc[pos + 18: pos + entry_len].rstrip(b"\x00").decode("utf-8", "replace")
    if typ.decode() in ("z", "Z"):
        pyz_raw = data[archive_start + entry_pos: archive_start + entry_pos + csize]
        print(f"PYZ at archive+{entry_pos}, csize={csize}, flag={cflag}")
    pos += entry_len

assert pyz_raw
print("\nfirst 96 bytes of PYZ:")
print(pyz_raw[:96].hex(" "))
print("ascii:", "".join(chr(b) if 32 <= b < 127 else "." for b in pyz_raw[:96]))

h = pyz_raw
print("\ninterpretations:")
print("  [4:8]  as !I :", struct.unpack("!I", h[4:8])[0], "(pyc magic?)")
print("  [8:12] as !I :", struct.unpack("!I", h[8:12])[0])
print("  [8:16] as !Q :", struct.unpack("!Q", h[8:16])[0])
print("  [12:16] as !I:", struct.unpack("!I", h[12:16])[0])
print("  [12:20] as !Q:", struct.unpack("!Q", h[12:20])[0])

# try marshal TOC at plausible offsets (4-byte tocpos -> toc at that offset)
for label, off in [("8:12 as !I toc", struct.unpack("!I", h[8:12])[0]),
                   ("12:16 as !I toc", struct.unpack("!I", h[12:16])[0]),
                   ("8:16 as !Q toc", struct.unpack("!Q", h[8:16])[0])]:
    if off == 0 or off >= len(h):
        print(f"{label}: invalid offset {off}")
        continue
    try:
        tl = marshal.loads(h[off:])
        print(f"{label}: OK -> {len(tl)} modules")
        names = sorted(n for n, _ in tl)
        app = [n for n in names if n.split(".")[0] in
               ("processors", "ui", "src", "models", "utils", "api")]
        print("app modules in PYZ:", len(app))
        for n in app[:60]:
            print("   ", n)
        break
    except Exception as e:
        print(f"{label}: fail ({e})")

# Also: scan entire PYZ for zlib streams and decompress a few to find module code objects
print("\n--- scanning for zlib-compressed code objects ---")
found = 0
app_names = set()
i = 16
while i < len(h) - 2 and found < 4000:
    if h[i] == 0x78 and h[i+1] in (0x01, 0x9c, 0xda, 0x5e):
        try:
            d = zlib.decompressobj()
            out = d.decompress(h[i:i + 400000])
            if len(out) > 50:
                try:
                    co = marshal.loads(out)
                    fname = getattr(co, "co_filename", "")
                    cname = getattr(co, "co_name", "")
                    if any(k in str(fname) for k in ("main_window", "image_processor", "app.py", "sequence", "loading_dialog")):
                        print(f"  zlib@{i}: code object file={fname} name={cname}")
                        app_names.add(str(fname))
                except Exception:
                    pass
                found += 1
                i += 10  # skip ahead
                continue
        except Exception:
            pass
    i += 1
print("zlib streams found:", found)
