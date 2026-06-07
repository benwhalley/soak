"""Security regression tests for document ingestion.

Covers two hardening measures in soak.document_utils:

1. safer_extract() zip-archive defences -- path traversal, symlinks, file-count
   cap, and total-uncompressed-size cap (zip bomb).
2. XXE resistance -- malicious DOCTYPE/external-entity payloads in HTML and
   office documents must never cause a local file read or network fetch.
"""

import os
import zipfile
from pathlib import Path

import pytest

from soak.document_utils import (
    DEFAULT_MAX_UNCOMPRESSED_ZIP_BYTES,
    ZipSecurityError,
    extract_text,
    safer_extract,
)

CANARY = "TOPSECRET_XXE_CANARY_8675309"


# --------------------------------------------------------------------------- #
# zip-bomb / unsafe-archive defences
# --------------------------------------------------------------------------- #


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_zip_uncompressed_size_cap(tmp_path, monkeypatch):
    """A zip whose uncompressed size exceeds the ceiling is refused before extract."""
    # highly compressible payload: ~1MB of zeros compresses to almost nothing
    big = b"\0" * (1024 * 1024)
    zpath = tmp_path / "bomb.zip"
    _make_zip(zpath, {"a.txt": big, "b.txt": big})

    monkeypatch.setenv("SOAK_MAX_UNCOMPRESSED_ZIP_BYTES", str(512 * 1024))
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        with pytest.raises(ZipSecurityError, match="uncompressed size exceeds"):
            safer_extract(zf, str(dest))
    # nothing should have been written
    assert not any(dest.iterdir())


def test_zip_size_cap_env_override_allows(tmp_path, monkeypatch):
    """Raising the env ceiling lets a previously-rejected archive through."""
    payload = b"\0" * (1024 * 1024)
    zpath = tmp_path / "ok.zip"
    _make_zip(zpath, {"a.txt": payload})

    monkeypatch.setenv("SOAK_MAX_UNCOMPRESSED_ZIP_BYTES", str(8 * 1024 * 1024))
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        safer_extract(zf, str(dest))
    assert (dest / "a.txt").read_bytes() == payload


def test_zip_default_ceiling_is_sane():
    assert DEFAULT_MAX_UNCOMPRESSED_ZIP_BYTES == 200 * 1024 * 1024


def test_zip_too_many_files(tmp_path):
    zpath = tmp_path / "many.zip"
    _make_zip(zpath, {f"f{i}.txt": b"x" for i in range(5)})
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        with pytest.raises(ZipSecurityError, match="too many files"):
            safer_extract(zf, str(dest), max_files=2)


def test_zip_path_traversal_blocked(tmp_path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../escape.txt", b"pwned")
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        with pytest.raises(ZipSecurityError, match="Unsafe path"):
            safer_extract(zf, str(dest))
    assert not (tmp_path / "escape.txt").exists()


def test_zip_symlink_blocked(tmp_path):
    zpath = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    # mark member as a symlink in the unix mode bits
    info.external_attr = (0o120777 & 0xFFFF) << 16
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(info, "/etc/passwd")
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        with pytest.raises(ZipSecurityError, match="Symlink"):
            safer_extract(zf, str(dest))


# --------------------------------------------------------------------------- #
# XXE resistance
# --------------------------------------------------------------------------- #


@pytest.fixture
def secret_file(tmp_path):
    p = tmp_path / "secret.txt"
    p.write_text(CANARY)
    return p


def _try_extract(path) -> str:
    """Return extracted text, or '' if the parser rejected the document.

    A parser that raises on a hostile DOCTYPE is a safe outcome -- the point of
    the test is that the canary never reaches the output (no file read happened).
    """
    try:
        result = extract_text(str(path))
    except Exception:
        return ""
    return result if isinstance(result, str) else str(result)


def test_html_external_entity_not_resolved(tmp_path, secret_file):
    html = (
        '<?xml version="1.0"?>\n'
        f'<!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file://{secret_file}"> ]>\n'
        "<html><body><p>start &xxe; end. Padding text so trafilatura has enough "
        "visible content to extract: lorem ipsum dolor sit amet consectetur "
        "adipiscing elit sed do eiusmod tempor incididunt ut labore.</p>"
        "</body></html>"
    )
    hp = tmp_path / "evil.html"
    hp.write_text(html)
    assert CANARY not in _try_extract(hp)


def test_docx_external_entity_not_resolved(tmp_path, secret_file):
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<!DOCTYPE w:document [ <!ENTITY xxe SYSTEM "file://{secret_file}"> ]>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>start &xxe; end</w:t></w:r></w:p></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'wordprocessingml.document.main+xml"/></Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    dp = tmp_path / "evil.docx"
    with zipfile.ZipFile(dp, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc_xml)
    assert CANARY not in _try_extract(dp)


def test_xlsx_external_entity_not_resolved(tmp_path, secret_file):
    """openpyxl must parse with defused XML so a shared-strings XXE is inert."""
    import openpyxl

    # build a clean workbook, then inject an XXE DOCTYPE into sharedStrings.xml
    wb = openpyxl.Workbook()
    wb.active["A1"] = "placeholder"
    clean = tmp_path / "clean.xlsx"
    wb.save(clean)

    evil = tmp_path / "evil.xlsx"
    shared = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<!DOCTYPE sst [ <!ENTITY xxe SYSTEM "file://{secret_file}"> ]>\n'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'count="1" uniqueCount="1"><si><t>start &xxe; end</t></si></sst>'
    )
    with zipfile.ZipFile(clean) as zin:
        names = zin.namelist()
        with zipfile.ZipFile(evil, "w") as zout:
            for n in names:
                if n == "xl/sharedStrings.xml":
                    continue
                zout.writestr(n, zin.read(n))
            zout.writestr("xl/sharedStrings.xml", shared)

    assert CANARY not in _try_extract(evil)


def test_defusedxml_is_active():
    """openpyxl reports defusedxml available -> its parsing is hardened."""
    from openpyxl import DEFUSEDXML

    assert DEFUSEDXML is True
