#!/bin/bash
# Builds "Install Joshua.app" and Install-Joshua.dmg (run on a Mac with the Command Line Tools).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build && mkdir -p "build/Install Joshua.app/Contents/MacOS" "build/Install Joshua.app/Contents/Resources" build/dmg
for arch in arm64 x86_64; do  # one app for Apple chips and Intel Macs
  swiftc -O -parse-as-library -target $arch-apple-macos13 InstallJoshua.swift -o build/InstallJoshua-$arch
done
lipo -create build/InstallJoshua-arm64 build/InstallJoshua-x86_64 -output "build/Install Joshua.app/Contents/MacOS/InstallJoshua"
cp Joshua.icns "build/Install Joshua.app/Contents/Resources/Joshua.icns"
cat > "build/Install Joshua.app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Install Joshua</string>
  <key>CFBundleDisplayName</key><string>Install Joshua</string>
  <key>CFBundleIdentifier</key><string>earth.joshua.installer</string>
  <key>CFBundleExecutable</key><string>InstallJoshua</string>
  <key>CFBundleIconFile</key><string>Joshua</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --deep -s - "build/Install Joshua.app"   # ad-hoc: not yet signed with an Apple Developer ID
cp -R "build/Install Joshua.app" build/dmg/
hdiutil create -volname "Install Joshua" -srcfolder build/dmg -ov -format UDZO build/Install-Joshua.dmg >/dev/null
ls -la build/Install-Joshua.dmg
