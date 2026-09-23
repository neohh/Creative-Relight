import os, sys, struct, marshal, zlib, subprocess

PY313 = r'C:\Users\maxim\AppData\Local\Programs\Python\Python313\python.exe'
ORIG_EXE = r'C:\Program Files\Creative Relight\Creative Relight.exe.bak_before_numbering'
OUT_EXE = r'C:\Users\maxim\Creative_Relight_patched.exe'

# Exact code for image_processor.py from 1:44 (when everything worked)
source_code = '''import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import sys
import re

# Handle imports for both development and frozen (PyInstaller) environments
try:
    from lighting_model import LightingModel
    from geometry_model import GeometryModel
except ModuleNotFoundError:
    from models.lighting_model import LightingModel
    from models.geometry_model import GeometryModel

try:
    from image_utils import resize_to_original, ensure_uint8
except ModuleNotFoundError:
    from utils.image_utils import resize_to_original, ensure_uint8

class ImageProcessor:
    """Process single images to generate all 5 passes"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lighting_model = LightingModel(device)
        self.geometry_model = GeometryModel(device)
        self.should_stop = False
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
    
    def process(self, image_input, output_dir, export_config=None, start_number=0, padding=6):
        """
        Process a single image and save selected passes
        """
        self.should_stop = False
        
        if export_config is None:
            export_config = {
                'albedo': True,
                'specular': True,
                'depth': True,
                'normal': True
            }
        
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input
        
        original_size = image.size
        image_array = np.array(image).astype(np.float32) / 255.0
        
        output_path = Path(output_dir)
        dirs = {}
        for component in ['albedo', 'specular', 'depth', 'normal']:
            if export_config.get(component, False):
                dirs[component] = output_path / component
                dirs[component].mkdir(parents=True, exist_ok=True)
        
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        lighting_results = {}
        if any(export_config.get(comp, False) for comp in ['albedo', 'specular']):
            lighting_results = self.lighting_model.process(image_array)
        
        if self.should_stop:
            return {'saved_files': {}, 'previews': {}, 'message': 'Processing stopped'}
        
        geometry_results = {}
        if any(export_config.get(comp, False) for comp in ['depth', 'normal']):
            geometry_results = self.geometry_model.process(image_array)
        
        saved_files = {}
        previews = {}
        
        # Auto-detect next free frame number by checking existing files
        max_existing = -1
        for comp, comp_dir in dirs.items():
            if comp_dir.exists():
                for item in comp_dir.iterdir():
                    m = re.match(rf"^{comp}_(\\d+)\\.png$", item.name)
                    if m:
                        max_existing = max(max_existing, int(m.group(1)))
        
        if max_existing >= 0:
            assigned_num = max_existing + 1
        else:
            assigned_num = int(start_number) if start_number is not None else 0
            
        frame_number = f"{assigned_num:0{padding}d}"
        
        # Save albedo
        if export_config.get('albedo', False) and 'albedo' in lighting_results and lighting_results['albedo'] is not None:
            albedo = resize_to_original(lighting_results['albedo'], original_size)
            albedo_uint8 = ensure_uint8(albedo)
            albedo_path = dirs['albedo'] / f"albedo_{frame_number}.png"
            Image.fromarray(albedo_uint8).save(albedo_path)
            saved_files['albedo'] = str(albedo_path)
            previews['albedo'] = albedo_uint8
        
        # Save specular
        if export_config.get('specular', False) and 'specular' in lighting_results and lighting_results['specular'] is not None:
            specular = resize_to_original(lighting_results['specular'], original_size)
            specular_uint8 = ensure_uint8(specular)
            specular_path = dirs['specular'] / f"specular_{frame_number}.png"
            if len(specular_uint8.shape) == 2:
                Image.fromarray(specular_uint8, mode='L').save(specular_path)
            else:
                Image.fromarray(specular_uint8).save(specular_path)
            saved_files['specular'] = str(specular_path)
            previews['specular'] = specular_uint8
        
        # Save depth
        if export_config.get('depth', False) and 'depth' in geometry_results and geometry_results['depth'] is not None:
            depth = cv2.resize(geometry_results['depth'], original_size, interpolation=cv2.INTER_LINEAR)
            depth_path = dirs['depth'] / f"depth_{frame_number}.png"
            cv2.imwrite(str(depth_path), depth)
            saved_files['depth'] = str(depth_path)
            previews['depth'] = depth
        
        # Save normal
        if export_config.get('normal', False) and 'normal' in geometry_results and geometry_results['normal'] is not None:
            normal = cv2.resize(geometry_results['normal'], original_size, interpolation=cv2.INTER_LINEAR)
            normal_path = dirs['normal'] / f"normal_{frame_number}.png"
            cv2.imwrite(str(normal_path), cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
            saved_files['normal'] = str(normal_path)
            previews['normal'] = normal
        
        return {
            'saved_files': saved_files,
            'previews': previews
        }
'''

temp_py = r'C:\Users\maxim\temp_image_processor.py'
with open(temp_py, 'w', encoding='utf-8') as f:
    f.write(source_code)

compile_runner = f"""
import marshal, zlib
with open(r'{temp_py}', 'r', encoding='utf-8') as f:
    code = compile(f.read(), 'src/processors/image_processor.py', 'exec')
compressed = zlib.compress(marshal.dumps(code, 4), 9)
with open(r'C:\\Users\\maxim\\compiled_blob.bin', 'wb') as f:
    f.write(compressed)
print('Successfully compiled! Size:', len(compressed))
"""
res = subprocess.run([PY313, '-c', compile_runner], capture_output=True, text=True)
print("Compiler STDOUT:", res.stdout.strip())
if res.stderr:
    print("Compiler STDERR:", res.stderr.strip())
if res.returncode != 0:
    print("Compilation failed!")
    sys.exit(1)

with open(r'C:\Users\maxim\compiled_blob.bin', 'rb') as f:
    new_compressed_code = f.read()

print("Reading base original exe:", ORIG_EXE)
with open(ORIG_EXE, 'rb') as f:
    exe_data = f.read()

magic = b'MEI\014\013\012\013\016'
cookie_pos = exe_data.rfind(magic)
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

# Patch ONLY image_processor, leave main_window 100% original
new_pyz_toc = []
found_target = False
new_entry_offset = old_pyz_toc_off
new_entry_len = len(new_compressed_code)

for item in pyz_toc:
    mod_name, (is_pkg, doff, dlen) = item
    if mod_name == 'src.processors.image_processor':
        found_target = True
        print(f"Updating PYZ entry for {mod_name}")
        new_pyz_toc.append((mod_name, (is_pkg, new_entry_offset, new_entry_len)))
    else:
        new_pyz_toc.append(item)

assert found_target, "src.processors.image_processor not found in PYZ TOC"

new_pyz_toc_off = new_entry_offset + new_entry_len

# Serialize with Python 3.13 runner to guarantee marshal format
serialize_runner = f"""
import marshal
with open(r'C:\\Users\\maxim\\toc_temp.bin', 'wb') as f:
    pass
"""
# Or directly with Python 3.13
with open(r'C:\Users\maxim\toc_temp.py', 'w', encoding='utf-8') as f:
    f.write(f"""
import marshal
toc = {repr(new_pyz_toc)}
with open(r'C:\\Users\\maxim\\marshalled_toc.bin', 'wb') as f:
    f.write(marshal.dumps(toc, 4))
""")

subprocess.run([PY313, r'C:\Users\maxim\toc_temp.py'], check=True)
with open(r'C:\Users\maxim\marshalled_toc.bin', 'rb') as f:
    marshalled_new_pyz_toc = f.read()

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

print(f"ROLLBACK BUILD COMPLETE: {OUT_EXE} ({len(new_exe_data)} bytes)")
print("main_window is 100% untouched original. ONLY image_processor auto-numbering is included.")
