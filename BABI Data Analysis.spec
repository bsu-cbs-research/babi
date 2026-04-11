# -*- mode: python ; coding: utf-8 -*-

import sys


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
    [],
    exclude_binaries=True,
    name='BABI Data Analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['program/static/program-exe-icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BABI Data Analysis',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='BABI Data Analysis.app',
        icon='program/static/program-exe-icon.ico',
        bundle_identifier=None,
    )
