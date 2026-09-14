"""Recursive structural comparison of frozen vs disk-compiled code objects.
Reports the specific functions that differ, ignoring filename/line-number tables."""
import struct, zlib, marshal, sys, types, dis, io

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXE = "repo/Creative Relight.exe"
with open(EXE, "rb") as f:
    data = f.read()

MAGIC = b"MEI\x0c\x0b\x0a\x0b\x0e"
idx = data.rfind(MAGIC)
magic, pkglen, toc_off, toclen, pyvers, pylib = struct.unpack("!8sIIII64s", data[idx: idx + 88])
archive_start = idx + 88 - pkglen
ctoc = data[archive_start + toc_off: archive_start + toc_off + toclen]

pos = 0
pyz_start = pyz_size = None
while pos < len(ctoc):
    (entry_len,) = struct.unpack("!i", ctoc[pos:pos + 4])
    entry_pos, csize, usize, cflag, typ = struct.unpack("!IIIBc", ctoc[pos + 4:pos + 18])
    name = ctoc[pos + 18: pos + entry_len].rstrip(b"\x00").decode()
    if typ.decode() in ("z", "Z"):
        pyz_start, pyz_size = entry_pos, csize
    pos += entry_len

pyz = data[archive_start + pyz_start: archive_start + pyz_start + pyz_size]
(pytocpos,) = struct.unpack("!I", pyz[8:12])
toc = dict(marshal.loads(pyz[pytocpos:]))

def strip(co):
    """Comparable signature of a code object ignoring positions/filename."""
    return (
        co.co_name, co.co_qualname, co.co_argcount, co.co_posonlyargcount,
        co.co_kwonlyargcount, co.co_nlocals, co.co_flags, co.co_code,
        tuple(strip(c) if isinstance(c, types.CodeType) else c for c in co.co_consts),
        co.co_names, co.co_varnames, co.co_freevars, co.co_cellvars,
    )

def walk(co, prefix=""):
    yield prefix + co.co_name, co
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from walk(c, prefix + co.co_name + ".")

MAP = {
    "src.processors.image_processor": "repo/src/processors/image_processor.py",
    "src.processors.sequence_processor": "repo/src/processors/sequence_processor.py",
    "src.processors.video_processor": "repo/src/processors/video_processor.py",
    "src.utils.file_handler": "repo/src/utils/file_handler.py",
    "src.utils.image_utils": "repo/src/utils/image_utils.py",
    "ui.main_window": "repo/ui/main_window.py",
    "ui.media_preview": "repo/ui/media_preview.py",
    "ui.processing_worker": "repo/ui/processing_worker.py",
}

for mod, path in MAP.items():
    if mod not in toc:
        print(f"=== {mod}: NOT IN PYZ ===")
        continue
    t, p, l = toc[mod]
    co_frozen = marshal.loads(zlib.decompress(pyz[p:p + l]))
    src = open(path, "rb").read()
    co_disk = compile(src, co_frozen.co_filename, "exec")

    frozen_map = dict(walk(co_frozen))
    disk_map = dict(walk(co_disk))

    only_frozen = set(frozen_map) - set(disk_map)
    only_disk = set(disk_map) - set(frozen_map)
    differing = [n for n in sorted(set(frozen_map) & set(disk_map))
                 if strip(frozen_map[n]) != strip(disk_map[n])]
    status = "IDENTICAL" if not (only_frozen or only_disk or differing) else "DIFFERS"
    print(f"=== {mod}: {status} (frozen={len(frozen_map)} code objects) ===")
    for n in differing[:25]:
        print(f"    differs: {n}")
    for n in sorted(only_frozen)[:10]:
        print(f"    only-frozen: {n}")
    for n in sorted(only_disk)[:10]:
        print(f"    only-disk: {n}")
