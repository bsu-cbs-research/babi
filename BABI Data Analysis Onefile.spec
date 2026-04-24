# -*- mode: python ; coding: utf-8 -*-

# Onefile build variant of "BABI Data Analysis.spec".
# Keep the Analysis() block in sync with the directory-bundle spec; the only
# differences should be the EXE() arguments below (single binary, no COLLECT).

a = Analysis(
    ['program/app.py'],
    pathex=['.'],
    binaries=[],
    datas=[('program/static', 'program/static')],
    hiddenimports=[
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'ai_edge_litert.interpreter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'ipykernel',
        'ipywidgets',
        'jupyterlab-widgets',
        'IPython',
        'matplotlib',
        'tensorflow',
        'keras',
        'tensorboard',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BABI Data Analysis Onefile',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['program/static/program-exe-icon.ico'],
)
