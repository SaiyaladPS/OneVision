#define AppName "Car Scan"
#define AppVersion "0.1.0"
#define AppPublisher "Car Scan"
#define AppExeName "CarScan.exe"

[Setup]
AppId={{B8F0F3FD-BB6A-4DA0-9E2E-4B2A3EAD9B21}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Car Scan
DefaultGroupName=Car Scan
OutputDir=output
OutputBaseFilename=CarScan-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin

[Files]
Source: "..\dist\CarScan\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Car Scan"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\Car Scan"; Filename: "{app}\{#AppExeName}"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "เปิด Car Scan"; Flags: nowait postinstall skipifsilent
