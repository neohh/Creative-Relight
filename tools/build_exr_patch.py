import os, sys, struct, marshal, zlib, subprocess

PY313 = r'C:\Users\maxim\AppData\Local\Programs\Python\Python313\python.exe'
ORIG_EXE = r'C:\Program Files\Creative Relight\Creative Relight.exe.bak_before_numbering'
OUT_EXE = r'C:\Users\maxim\Creative_Relight_patched.exe'

temp_ip = r'C:\Users\maxim\temp_image_processor.py'

print("[1] Compiling updated image_processor.py with Python 3.13...")
with open(temp_ip, 'r', encoding='utf-8') as f:
    src_code = f.read()

code_obj = compile(src_code, 'src/processors/image_processor.py', 'exec')
new_compressed_code = zlib.compress(marshal.dumps(code_obj, 4), 9)
print(f"  Compressed blob size: {len(new_compressed_code)} bytes")

print(f"[2] Reading base original exe: {ORIG_EXE}...")
with open(ORIG_EXE, 'rb') as f:
    exe_data = f.read()

magic = b'MEI\014\013\012\013\016'
cookie_pos = exe_data.rfind(magic)
if cookie_pos == -1:
    raise ValueError("PyInstaller cookie not found!")

cookie = exe_data[cookie_pos:cookie_pos+88]
pkg_size, old_toc_off, old_toc_size, pyver = struct.unpack('!IIII', cookie[8:24])
pylib = cookie[24:88]
pkg_start = len(exe_data) - pkg_size
exe_head = exe_data[:pkg_start]

old_toc_abs = pkg_start + old_toc_off
old_toc_data = exe_data[old_toc_abs:old_toc_abs+old_toc_size]

pos = 0
pkg_entries = []
while pos < len(old_toc_data):
    if pos + 18 > len(old_toc_data): break
    elen = struct.unpack('!I', old_toc_data[pos:pos+4])[0]
    doff = struct.unpack('!I', old_toc_data[pos+4:pos+8])[0]
    dlen = struct.unpack('!I', old_toc_data[pos+8:pos+12])[0]
    ulen = struct.unpack('!I', old_toc_data[pos+12:pos+16])[0]
    flag = old_toc_data[pos+16]
    tc = chr(old_toc_data[pos+17])
    name = old_toc_data[pos+18:pos+elen].split(b'\x00')[0].decode('utf-8', errors='replace')
    pkg_entries.append({
        'name': name, 'tc': tc, 'doff': doff, 'dlen': dlen, 'ulen': ulen, 'flag': flag, 'elen': elen
    })
    pos += elen

pyz_entry = [e for e in pkg_entries if e['name'] == 'PYZ.pyz'][0]
pyz_doff = pyz_entry['doff']
pyz_abs = pkg_start + pyz_doff

pyz_header = exe_data[pyz_abs:pyz_abs+12]
pyz_magic = pyz_header[:4]
py_magic = pyz_header[4:8]
old_pyz_toc_off = struct.unpack('!I', pyz_header[8:12])[0]

pyz_data_before_toc = exe_data[pyz_abs:pyz_abs+old_pyz_toc_off]
old_pyz_toc_bytes = exe_data[pyz_abs+old_pyz_toc_off : old_toc_abs]
pyz_toc = marshal.loads(old_pyz_toc_bytes)

new_pyz_toc = []
found_target = False
new_entry_offset = old_pyz_toc_off
new_entry_len = len(new_compressed_code)

for item in pyz_toc:
    mod_name, (is_pkg, doff, dlen) = item
    if mod_name == 'src.processors.image_processor':
        found_target = True
        print(f"  [OK] Patching PYZ entry for {mod_name} (offset={new_entry_offset}, len={new_entry_len})")
        new_pyz_toc.append((mod_name, (is_pkg, new_entry_offset, new_entry_len)))
    else:
        new_pyz_toc.append(item)

assert found_target, "src.processors.image_processor not found in PYZ TOC"

new_pyz_toc_off = new_entry_offset + new_entry_len

# Serialize TOC strictly with Python 3.13 marshal (version 4)
marshalled_new_pyz_toc = marshal.dumps(new_pyz_toc, 4)

new_pyz_header = pyz_magic + py_magic + struct.pack('!I', new_pyz_toc_off)
new_pyz = new_pyz_header + pyz_data_before_toc[12:] + new_compressed_code + marshalled_new_pyz_toc
new_pyz_len = len(new_pyz)

pkg_before_pyz = exe_data[pkg_start:pyz_abs]

new_pkg_toc_entries = bytearray()
for e in pkg_entries:
    if e['name'] == 'PYZ.pyz':
        dlen = new_pyz_len
        ulen = new_pyz_len
    else:
        dlen = e['dlen']
        ulen = e['ulen']
    name_bytes = e['name'].encode('utf-8') + b'\x00'
    elen = e['elen']
    raw = struct.pack('!IIIIBB', elen, e['doff'], dlen, ulen, e['flag'], ord(e['tc'])) + name_bytes
    if len(raw) < elen:
        raw += b'\x00' * (elen - len(raw))
    new_pkg_toc_entries.extend(raw)

new_pkg_toc_bytes = bytes(new_pkg_toc_entries)

new_toc_off = pyz_doff + new_pyz_len
new_toc_size = len(new_pkg_toc_bytes)
new_cookie = magic + struct.pack('!IIII', new_toc_off + new_toc_size + 88, new_toc_off, new_toc_size, pyver) + pylib

new_exe_data = exe_head + pkg_before_pyz + new_pyz + new_pkg_toc_bytes + new_cookie

with open(OUT_EXE, 'wb') as f:
    f.write(new_exe_data)

print(f"[3] Saved updated executable to: {OUT_EXE} ({len(new_exe_data)} bytes)")

# Self-verification
print("[4] Verifying newly built executable...")
with open(OUT_EXE, 'rb') as f:
    v_data = f.read()

v_cookie_pos = v_data.rfind(magic)
v_cookie = v_data[v_cookie_pos:v_cookie_pos+88]
v_pkg_size, v_toc_off, v_toc_size, v_pyver = struct.unpack('!IIII', v_cookie[8:24])
v_pkg_start = len(v_data) - v_pkg_size
v_pyz_abs = v_pkg_start + pyz_doff
v_pyz_hdr = v_data[v_pyz_abs:v_pyz_abs+12]
v_toc_off_pyz = struct.unpack('!I', v_pyz_hdr[8:12])[0]
v_pyz_toc_bytes = v_data[v_pyz_abs + v_toc_off_pyz : v_pkg_start + v_toc_off]
v_toc = marshal.loads(v_pyz_toc_bytes)

ip_entry = [x for x in v_toc if x[0] == 'src.processors.image_processor'][0]
ip_raw = v_data[v_pyz_abs + ip_entry[1][1] : v_pyz_abs + ip_entry[1][1] + ip_entry[1][2]]
ip_code = marshal.loads(zlib.decompress(ip_raw))
print(f"  image_processor verified! Code name: {ip_code.co_name}, consts: {len(ip_code.co_consts)}")

print("\nSUCCESS: 32-bit EXR build ready!")
