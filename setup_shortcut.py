"""
tiser — Desktop-Verknuepfung erstellen (einmalig aus CMD ausfuehren)
"""
import sys, os, subprocess

PYTHON  = sys.executable
SCRIPT  = r"C:\tiser\app.py" # Pfad entsprechend auf den Desktop-Pfad anpassen
WORKDIR = r"C:\tiser"

print(f"Python:  {PYTHON}")
print(f"Skript:  {SCRIPT}")
print()

# PowerShell bestimmt den echten Desktop-Pfad selbst
# (funktioniert auch bei umgeleiteten Desktops auf Netzlaufwerken)
ps = (
    "$Desktop = [Environment]::GetFolderPath('Desktop'); "
    "$LNK = Join-Path $Desktop 'tiser.lnk'; "
    f"$ws = New-Object -ComObject WScript.Shell; "
    "$sc = $ws.CreateShortcut($LNK); "
    f"$sc.TargetPath = '{PYTHON}'; "
    f"$sc.Arguments = '\"{SCRIPT}\"'; "
    f"$sc.WorkingDirectory = '{WORKDIR}'; "
    "$sc.WindowStyle = 1; "
    "$sc.Description = 'starten'; "
    "$sc.Save(); "
    "Write-Host $LNK"
)

# Kein capture_output: PowerShell schreibt direkt ins CMD-Fenster
# Vermeidet den Encoding-Fehler und zeigt echte Fehlermeldungen
result = subprocess.run(
    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps]
)

print()
if result.returncode == 0:
    print("OK  Verknuepfung erstellt (Pfad steht oben).")
    print("    Doppelklick auf 'tiser' startet den Server und oeffnet Edge.")
else:
    print(f"PowerShell-Fehler (Code {result.returncode}).")
    print()
    print("Manuell erstellen:")
    print("  Rechtsklick auf Desktop -> Neu -> Verknuepfung")
    print(f"  Ziel:    {PYTHON}")
    print(f"  Argumen: \"{SCRIPT}\"")

input("\nEnter druecken...")
