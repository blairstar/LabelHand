# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['LabelHand\\__main__.py'],
             pathex=[],
             binaries=[],
             datas=[('./LabelHand/config', 'LabelHand/config'), ('./LabelHand/translate', 'LabelHand/translate'), ('D:\\ProgramFiles\\Anaconda3\\envs\\py36\\Lib\\site-packages\\open3d\\resources', 'open3d\\resources'), ('./Model', 'Model'), ('./Template', 'Template'), ('./Test', 'Test'), ('./doc', 'doc')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
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
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='LabelHand')
