# -*- mode: python -*-

block_cipher = None


a = Analysis(['D:\\codes\\WeSee\\LabelHand\\LabelHand\\src\\main\\python\\main.py'],
             pathex=['D:\\codes\\WeSee\\LabelHand\\LabelHand\\target\\PyInstaller'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=['d:\\programfiles\\anaconda3\\envs\\py36\\lib\\site-packages\\fbs\\freeze\\hooks'],
             runtime_hooks=['D:\\codes\\WeSee\\LabelHand\\LabelHand\\target\\PyInstaller\\fbs_pyinstaller_hook.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='LabelHand',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=False , version='D:\\codes\\WeSee\\LabelHand\\LabelHand\\target\\PyInstaller\\version_info.py', icon='D:\\codes\\WeSee\\LabelHand\\LabelHand\\src\\main\\icons\\Icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='LabelHand')
