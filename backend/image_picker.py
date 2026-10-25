"""Open the system file window and read the chosen images.

A pinch is not a browser click, so the page cannot open ``<input type="file">``
itself. The bridge opens the native dialog instead and sends the bytes back.
"""

from __future__ import annotations

import base64
import mimetypes
import subprocess
import sys
from pathlib import Path

CANCELLED = "__CANCELLED__"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_DARWIN_SCRIPT = r'''
try
  set chosen to choose file with prompt "Add images" of type {"public.image"} with multiple selections allowed
on error errMsg number errNum
  if errNum is -128 then
    return "__CANCELLED__"
  end if
  error errMsg number errNum
end try
set output to ""
repeat with f in chosen
  set output to output & POSIX path of f & linefeed
end repeat
return output
'''

_WINDOWS_SCRIPT = r'''
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Multiselect = $true
$dialog.Title = "Add images"
$dialog.Filter = "Images|*.png;*.jpg;*.jpeg;*.webp;*.gif;*.bmp"
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$result = $dialog.ShowDialog($owner)
$owner.Dispose()
if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
  Write-Output "__CANCELLED__"
  exit 0
}
$dialog.FileNames -join "`n"
'''


def paths_from_dialog_output(text: str) -> list[Path] | None:
    """``None`` means the person closed the window without choosing."""
    cleaned = (text or "").strip()
    if not cleaned or cleaned == CANCELLED:
        return None
    paths: list[Path] = []
    for line in cleaned.splitlines():
        raw = line.strip()
        if raw:
            paths.append(Path(raw))
    return paths or None


def pick_image_paths() -> list[Path] | None:
    """Show the native multi-image dialog. ``None`` means it was cancelled."""
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["osascript", "-e", _DARWIN_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    elif sys.platform == "win32":
        completed = subprocess.run(
            ["powershell", "-STA", "-NoProfile", "-Command", _WINDOWS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    else:
        completed = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--multiple",
                "--separator",
                "\n",
                "--title",
                "Add images",
                "--file-filter",
                "Images | *.png *.jpg *.jpeg *.webp *.gif *.bmp",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if completed.returncode != 0:
            return None
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "Could not open the file window.").strip()
        raise RuntimeError(message)
    return paths_from_dialog_output(completed.stdout)


def encode_images(paths: list[Path]) -> list[dict[str, str]]:
    """Read chosen images into JSON-ready records. Other files are skipped."""
    encoded: list[dict[str, str]] = []
    for path in paths:
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        data = path.read_bytes()
        if not data or len(data) > MAX_IMAGE_BYTES:
            continue
        mime = IMAGE_MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded.append(
            {
                "name": path.name,
                "type": mime,
                "data_base64": base64.b64encode(data).decode("ascii"),
            }
        )
    return encoded
