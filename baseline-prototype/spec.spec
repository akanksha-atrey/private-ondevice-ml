# Import required PyInstaller modules
from PyInstaller.utils.hooks import collect_data_files

# Define the main script file
deployment_script = 'src/deployment.py'

# Define the list of additional data files
added_files = [ ('data/input.txt', 'data'),
                ('models/*.onnx', 'models')
                ]

# Configure PyInstaller
a = Analysis(
    [deployment_script],
    pathex = ['.'],
    binaries=[],
    datas=added_files,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False)

# Add any additional PyInstaller configuration as needed

# Create the executable using PyInstaller
pyz = PYZ(a.pure, a.zipped_data,
          cipher=None)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='baseline',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None)