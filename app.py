# -*- coding: utf-8 -*-
"""
tiseR - RAG-Backend   [Version 20260702v01] (AppLocker-konform, KEINE nativen DLLs)
Optimierungen ggue. v1:
  1. Strukturbewusstes Chunking (Ueberschriften-Schnitte, ToC/Kopfzeilen entfernt,
     Ueberschrift als Praefix im Chunk).
  2. Hybrid-Retrieval: Pure-Python BM25 (numpy-frei) + Cosine, fusioniert via
     Reciprocal Rank Fusion (RRF).
  3. PDF-Extraktion im LAYOUT-Modus (Tabellenspalten bleiben grob ausgerichtet).
  4. Zusaetzliche Formate DOCX + XLSX, geparst via stdlib zipfile + ElementTree
     (KEINE nativen DLLs -> AppLocker-konform; KEIN python-docx/openpyxl/lxml).
  5. /api/documents (Liste) + /api/delete_doc (Einzelloeschung) fuer die UI.
  6. Leichtgewichtige Metadaten-Tags pro Chunk (frist/schwellenwert/zustaendigkeit),
     per Regex beim Indexieren erkannt, SEPARAT gespeichert (nie ins Embedding-Fenster!)
     und fuer einen weichen, multiplikativen Boost in hybrid_search nutzbar.
     WICHTIG: Boost ist per Default NEUTRAL (TAG_BOOST = 1.0). Erst scharf schalten,
     wenn der Eval-Harness Hybrid+Boost messen kann — blind aktivieren waere falsch.
  7. Markdown (.md/.markdown) als Eingabeformat — inkl. YAML-Frontmatter (OKF-tauglich).
     KEIN Konverter: andere Formate werden NICHT nach Markdown gewandelt, nur native
     .md-Dateien direkt gelesen. Frontmatter-Tags fliessen in die Chunk-Tags ein.
PDF-Extraktion: pypdf (reines Python).
"""
import io, json, math, re, sqlite3, logging, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, g, abort

try:
    from pypdf import PdfReader
    PDF_OK = True
except Exception as _e:
    PDF_OK, _PDF_ERR = False, str(_e)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("armachat")

APP_DIR    = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODELS_DIR = APP_DIR / "models"
DATA_DIR   = APP_DIR / "data"
DB_PATH    = APP_DIR / "armachat.db"
HOST, PORT = "127.0.0.1", 8000

CHUNK_TARGET = 500      # Ziel-Zeichen/Chunk: passt ins Embedding-Fenster (~128 Tokens)
CHUNK_HARD   = 750      # harte Obergrenze
CHUNK_MIN    = 15       # Mindestlaenge (sonst gehen kurze Folien beim Seiten-Flush verloren)
TOP_K        = 4      # 3-5 Quellen reichen; gezielte Antwort steckt nicht in 6
RRF_K        = 60       # RRF-Konstante (Standardwert)
PAGE_BREAK   = "\x0c"   # Seiten-/Foliengrenze -> harter Chunk-Schnitt
# Metadaten-Boost: weicher, MULTIPLIKATIVER Faktor auf Chunks, deren Tags zum
# Fragetyp passen. 1.0 = AUS (Feature verdrahtet, aber wirkungslos). Erst auf
# z.B. 1.15 erhoehen, NACHDEM eval_harness.py Hybrid+Boost gegen reines BM25
# gemessen hat. Vorher waere jede Zahl reines Bauchgefuehl.
TAG_BOOST    = 1.0

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

VECTORS = []            # [{id, source, text, vec, toks}]
BM25 = None             # PureBM25 ueber VECTORS

@app.after_request
def isolation_headers(resp):
    resp.headers["Cross-Origin-Opener-Policy"]  = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    resp.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return resp

# --------------------------------------------------------------- DB
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # FIX: ohne dies sind Zeilen Tupel -> r["id"] crasht bei gefuellter DB
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, source TEXT NOT NULL, ord INTEGER NOT NULL,
        text TEXT NOT NULL, vec TEXT, tags TEXT)""")
    # Migration bestehender DBs (vor dem Tags-Feature angelegt): Spalte nachruesten.
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN tags TEXT")
    except sqlite3.OperationalError:
        pass   # Spalte existiert bereits -> nichts zu tun
    conn.commit()
    VECTORS.clear()
    for r in conn.execute("SELECT id, source, text, vec, tags FROM chunks WHERE vec IS NOT NULL"):
        VECTORS.append({"id": r["id"], "source": r["source"], "text": r["text"],
                        "vec": json.loads(r["vec"]), "toks": _tok(r["text"]),
                        "tags": json.loads(r["tags"]) if r["tags"] else []})
    conn.close()
    rebuild_bm25()
    log.info("DB bereit. %d Chunks mit Vektor.", len(VECTORS))

# --------------------------------------------------------------- PDF (Standard-Extraktion)
def extract_pdf_text(src) -> str:
    if not PDF_OK:
        raise RuntimeError(f"pypdf nicht geladen: {_PDF_ERR}")
    reader = PdfReader(src); parts = []
    for page in reader.pages:
        t = ""
        # Standard-Extraktion haelt Label+Wert zusammen ("Angefragte Mitarbeitende
        # > 1320"). Der frueher genutzte Layout-Modus liest seitenbreit zeilenweise
        # und verschraenkt dadurch raeumlich getrennte Bloecke (z.B. Folien mit
        # Text links + Diagramm rechts) — die Zahl landet zwischen fremden Werten
        # und verliert ihren Bezug. Layout zerbricht zudem Woerter an Buchstaben-
        # abstaenden ("Rem arks"). Daher Standard zuerst, Layout nur als Notnagel,
        # falls Standard fuer eine Seite gar nichts liefert.
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if not t.strip():
            try: t = page.extract_text(extraction_mode="layout") or ""
            except Exception: t = ""
        if t.strip(): parts.append(t)
    # Seitengrenzen als harte Schnittpunkte erhalten (wichtig bei Foliensaetzen)
    return ("\n" + PAGE_BREAK + "\n").join(parts)

# --------------------------------------------------------------- DOCX / XLSX (stdlib, keine DLLs)
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

def _docx_para_text(p):
    return "".join(t.text or "" for t in p.iter(_W + "t"))

def extract_docx_text(src) -> str:
    """Word: Absaetze als Zeilen, Tabellenzeilen als 'Zelle | Zelle | …'."""
    with zipfile.ZipFile(src) as z:
        with z.open("word/document.xml") as f:
            root = ET.parse(f).getroot()
    body = root.find(_W + "body")
    if body is None:
        return ""
    out = []
    for el in body:
        if el.tag == _W + "p":
            txt = _docx_para_text(el).strip()
            if txt: out.append(txt)
        elif el.tag == _W + "tbl":
            for tr in el.findall(_W + "tr"):
                cells = []
                for tc in tr.findall(_W + "tc"):
                    cell = " ".join(_docx_para_text(p).strip() for p in tc.findall(_W + "p"))
                    cells.append(cell.strip())
                row = " | ".join(c for c in cells if c)
                if row.strip(" |"): out.append(row)
    return "\n".join(out)

def extract_xlsx_text(src) -> str:
    """Excel: jede Zeile als 'Zelle | Zelle | …'; Shared-Strings aufgeloest."""
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            with z.open("xl/sharedStrings.xml") as f:
                sroot = ET.parse(f).getroot()
            for si in sroot.iter(_S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(_S + "t")))
        out = []
        for sn in sorted(n for n in names
                         if n.startswith("xl/worksheets/") and n.endswith(".xml")):
            if out: out.append(PAGE_BREAK)         # Blattgrenze = harter Schnitt
            with z.open(sn) as f:
                wroot = ET.parse(f).getroot()
            for row in wroot.iter(_S + "row"):
                vals = []
                for c in row.iter(_S + "c"):
                    typ = c.get("t"); v = c.find(_S + "v")
                    if typ == "s" and v is not None and v.text is not None:
                        try: vals.append(shared[int(v.text)])
                        except (ValueError, IndexError): pass
                    elif typ == "inlineStr":
                        isn = c.find(_S + "is")
                        if isn is not None:
                            vals.append("".join(x.text or "" for x in isn.iter(_S + "t")))
                    elif v is not None and v.text is not None:
                        vals.append(v.text)
                line = " | ".join(s for s in (str(x).strip() for x in vals) if s)
                if line: out.append(line)
    return "\n".join(out)

# --------------------------------------------------------------- PPTX (stdlib, eine Folie = ein Abschnitt)
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

def extract_pptx_text(src) -> str:
    """PowerPoint: Text je Folie (a:t-Elemente), Folien als harte Schnitte."""
    def _slide_no(n):
        m = re.search(r'slide(\d+)\.xml$', n); return int(m.group(1)) if m else 0
    with zipfile.ZipFile(src) as z:
        slides = sorted((n for n in z.namelist()
                         if re.match(r'ppt/slides/slide\d+\.xml$', n)), key=_slide_no)
        out = []
        for sn in slides:
            try:
                with z.open(sn) as f:
                    root = ET.parse(f).getroot()
            except Exception:
                continue                              # eine kaputte Folie kippt nicht den ganzen Satz
            texts = [t.text for t in root.iter(_A + "t") if t.text and t.text.strip()]
            body = " ".join(texts).strip()
            if body:
                if out: out.append(PAGE_BREAK)        # Foliengrenze
                out.append(body)
    return "\n".join(out)

# --------------------------------------------------------------- Plaintext-artige Formate (stdlib)
def _read_bytes(src) -> bytes:
    return src.read() if hasattr(src, "read") else Path(src).read_bytes()

def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: continue
    return raw.decode("utf-8", "ignore")

def extract_txt_text(src) -> str:
    return _decode(_read_bytes(src))

def extract_csv_text(src) -> str:
    import csv, io as _io
    text = _decode(_read_bytes(src))
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except Exception:
        dialect = csv.excel
    rows = csv.reader(_io.StringIO(text), dialect)
    out = []
    for r in rows:
        line = " | ".join(c.strip() for c in r if c and c.strip())
        if line: out.append(line)
    return "\n".join(out)

def _sender_label(raw: str) -> str:
    """Lesbarer Absendername aus einem From-Header.
    'Brechbühl Fabian <f.b@vbs.admin.ch>' -> 'Brechbühl Fabian'.
    Fallback: lokaler Teil der Adresse."""
    if not raw: return ""
    raw = raw.strip()
    m = re.match(r'\s*"?([^"<]+?)"?\s*<[^>]+>\s*$', raw)
    if m:
        name = m.group(1).strip().strip('"\'')
        if name: return name
    m = re.search(r'([\w.\-]+)@', raw)
    if m: return m.group(1).replace(".", " ")
    return raw

def extract_eml_text(src) -> str:
    """Standard-E-Mail (RFC 822) via stdlib email."""
    import email
    from email import policy
    msg = email.message_from_bytes(_read_bytes(src), policy=policy.default)
    head = []
    for h in ("From", "To", "Cc", "Subject", "Date"):
        v = msg.get(h)
        if v: head.append(f"{h}: {v}")
    body = ""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            body = part.get_content()
            if part.get_content_type() == "text/html":
                body = re.sub(r"<[^>]+>", " ", body)
    except Exception:
        body = ""
    # Absender als Markdown-Ueberschrift: der Chunker erkennt sie als 'head' und
    # praefixt JEDEN Chunk dieser Mail -> "Ich" im Body bleibt ueber alle Chunks
    # auf den Absender aufloesbar (Koreferenz-Fix, z.B. "Ich bin der EPIC-Owner").
    sender_short = _sender_label(msg.get("From") or "")
    md_head = [f"## Mail von {sender_short}"] if sender_short else []
    return "\n".join(md_head + head + ["", body.strip()])

# --------------------------------------------------------------- MSG (Outlook, binaeres OLE -> olefile, rein Python)
try:
    import olefile
    MSG_OK = True
except Exception as _me:
    MSG_OK, _MSG_ERR = False, str(_me)

def _msg_prop(ole, tag, prefix="") -> str:
    """Liest eine MAPI-Property (tag z.B. '1000') als Unicode (001F) o. ASCII (001E)."""
    for suffix, dec in (("001F", "utf-16-le"), ("001E", "cp1252")):
        name = prefix + "__substg1.0_" + tag + suffix
        if ole.exists(name):
            try: return ole.openstream(name).read().decode(dec, "ignore").strip()
            except Exception: return ""
    return ""

def extract_msg_text(src) -> str:
    if not MSG_OK:
        raise RuntimeError("Fuer .msg wird das reine-Python-Modul 'olefile' benoetigt "
                           "(offline installierbar, AppLocker-konform). Aktuell nicht geladen: " + _MSG_ERR)
    ole = olefile.OleFileIO(src)
    try:
        subject = _msg_prop(ole, "0037")                       # PR_SUBJECT
        sender  = _msg_prop(ole, "0C1A") or _msg_prop(ole, "0C1F")  # Name / E-Mail
        body    = _msg_prop(ole, "1000")                       # PR_BODY (Plaintext)
        if not body:
            html = _msg_prop(ole, "1013")                      # PR_HTML
            if html: body = re.sub(r"<[^>]+>", " ", html)
        recips = []
        for entry in ole.listdir():
            if entry and entry[0].startswith("__recip_version1.0_"):
                nm = _msg_prop(ole, "3001", prefix=entry[0] + "/")  # PR_DISPLAY_NAME
                if nm: recips.append(nm)
        sender_short = _sender_label(sender)
        md_head = [f"## Mail von {sender_short}"] if sender_short else []
        head = []
        if sender:  head.append("Von: " + sender)
        if recips:  head.append("An: " + ", ".join(dict.fromkeys(recips)))
        if subject: head.append("Betreff: " + subject)
        return "\n".join(md_head + head + ["", body.strip()])
    finally:
        ole.close()

# --------------------------------------------------------------- Markdown (.md, OKF-tauglich)
# KEIN Konverter: wir wandeln NICHTS nach Markdown. Wir LESEN nur native .md-Dateien.
# YAML-Frontmatter wird minimal (ohne PyYAML, reine stdlib) geparst: nur die flachen
# Felder, die OKF nutzt (title, tags). Reicht fuer OKF-Bundles und Confluence-Exporte.
_FM_DELIM = re.compile(r'^---\s*$')

def _parse_frontmatter(raw: str):
    """Trennt YAML-Frontmatter vom Body. Liefert (body, title, tags).
    Bewusst genuegsam: 'key: value' und 'tags: [a, b]' bzw. 'tags: a, b'.
    Verschachteltes YAML wird ignoriert (kommt in OKF-Frontmatter praktisch nicht vor)."""
    lines = raw.split("\n")
    if not lines or not _FM_DELIM.match(lines[0]):
        return raw, "", []           # kein Frontmatter -> alles ist Body
    end = None
    for i in range(1, len(lines)):
        if _FM_DELIM.match(lines[i]):
            end = i; break
    if end is None:
        return raw, "", []           # oeffnendes '---' ohne Abschluss -> als Body behandeln
    title, tags = "", []
    for ln in lines[1:end]:
        m = re.match(r'\s*([A-Za-z_][\w-]*)\s*:\s*(.*)$', ln)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "title":
            title = val.strip('"\'')
        elif key == "tags":
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                val = val[1:-1]
            tags = [t.strip().strip('"\'').lower() for t in val.split(",") if t.strip()]
    body = "\n".join(lines[end + 1:])
    return body, title, tags

def extract_md_text(src) -> str:
    """Markdown-Body als Text. Frontmatter-title wird als Markdown-Ueberschrift
    vorangestellt (echte Struktur -> wird vom Chunker als Praefix erkannt).
    Frontmatter-tags werden NICHT hier verarbeitet, sondern in extract_text_any
    (das den Tag-Kanal getrennt vom Text fuehrt)."""
    body, title, _tags = _parse_frontmatter(_decode(_read_bytes(src)))
    return (f"# {title}\n{body}") if title else body

def _md_seed_tags(src) -> list:
    """Liefert NUR die Frontmatter-tags (separater Kanal, geht nie ins Embedding)."""
    _body, _title, tags = _parse_frontmatter(_decode(_read_bytes(src)))
    return tags

# Dispatcher nach Dateiendung. PdfReader/ZipFile/olefile akzeptieren Pfad ODER BytesIO.
_EXT_HANDLERS = {
    ".pdf": extract_pdf_text,   ".docx": extract_docx_text, ".xlsx": extract_xlsx_text,
    ".pptx": extract_pptx_text, ".csv": extract_csv_text,   ".txt": extract_txt_text,
    ".eml": extract_eml_text,   ".msg": extract_msg_text,
    ".md": extract_md_text,     ".markdown": extract_md_text,
}
SUPPORTED_EXT = tuple(_EXT_HANDLERS.keys())

def extract_text_any(name: str, src):
    """Liefert (text, seed_tags). seed_tags stammen NUR aus .md-Frontmatter und
    sind ein vom Text getrennter Metadaten-Kanal (gelangen nie ins Embedding)."""
    ext = Path(name).suffix.lower()
    handler = _EXT_HANDLERS.get(ext)
    if not handler:
        raise ValueError(f"Format '{ext or '?'}' nicht unterstuetzt "
                         f"(PDF, DOCX, XLSX, PPTX, CSV, TXT, EML, MSG, MD)")
    # .md kann BytesIO/Pfad sein; seed_tags brauchen einen zweiten Lesedurchgang.
    # Bei BytesIO daher vor dem Handler-Aufruf die Tags ziehen und zuruecksetzen.
    seed_tags = []
    if ext in (".md", ".markdown"):
        if hasattr(src, "read"):
            seed_tags = _md_seed_tags(src)
            src.seek(0)                      # Stream fuer den Handler zuruecksetzen
        else:
            seed_tags = _md_seed_tags(src)
    return handler(src), seed_tags

# --------------------------------------------------------------- Metadaten-Tagger (heuristisch)
# Pure Python, AppLocker-safe. Bewusst konservativ. Tags landen SEPARAT (nie im Text).
_RX_FRIST    = re.compile(r'\b(frist(?:en)?|innerhalb|binnen|spätestens|spaetestens|'
                          r'werktag\w*|arbeitstag\w*|stunden?|tagen?|wochen?|monaten?)\b', re.I)
_RX_ZUST     = re.compile(r'\b(zuständig\w*|zustaendig\w*|verantwortlich\w*|obliegt|'
                          r'verantwortliche stelle|federführ\w*|federfuehr\w*)\b', re.I)
# Schwellenwert ist der UNZUVERLAESSIGSTE Tagger: nur Betraege/Vergleiche, KEINE
# nackten Zahlen (sonst feuert jede Dok-ID/Jahreszahl). 20'000 / CHF 5000 / ≥ 50 / > 1320
_RX_SCHWELLE = re.compile(r"(chf\s*\d|franken|schwellenwert\w*|\d['’]\d{3}|"
                          r"[≥≤<>]\s*\d|mindestens|höchstens|hoechstens)", re.I)

def _tag_chunk(text: str):
    """Erkennt Fragetyp-/Inhaltstyp-Tags. Reihenfolge stabil, fuer reproduzierbare Tests."""
    tags = []
    if _RX_FRIST.search(text):    tags.append("frist")
    if _RX_SCHWELLE.search(text): tags.append("schwellenwert")
    if _RX_ZUST.search(text):     tags.append("zustaendigkeit")
    return tags

# --------------------------------------------------------------- Strukturbewusstes Chunking
_TOC  = re.compile(r'\.{5,}\s*\d+\s*$')
# Abschnittsnummern wie "2.1.3" — Komponenten max. 2-stellig, schliesst Daten
# (z. B. "12.05.2026") und 4-stellige Jahre aus.
_HEAD = re.compile(r'^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-ZÄÖÜ].{2,80})$')
# Markdown-ATX-Ueberschrift ("## Titel"); macht .md-Eingaben erststklassig.
_MD_HEAD = re.compile(r'^(#{1,6})\s+(.+?)\s*#*$')
_HDR  = re.compile(r'^(MS ID/Ver|Dok-ID/Vers)\s')
# Wiederkehrende Folien-/Seitenfusszeilen (Datum, Copyright+Folie, reine Foliennr.)
_MONTHS = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
_FOOT = re.compile(
    r'^(?:\d{1,2}\.\s*(?:%s)\s+\d{4}'        # "8. Juni 2026"
    r'|©.*Folie\s*\d+.*'                       # "© APP … Folie 4"
    r'|Folie\s*\d+'                            # "Folie 4"
    r'|©\s*\S.*(?:AG|GmbH))\s*$' % _MONTHS)

def _clean_lines(text: str):
    text = text.replace("\xad", "")
    text = re.sub(r"-\s*\n\s*", "", text)          # Silbentrennung zusammenfuehren
    out = []
    for ln in text.split("\n"):
        if ln == PAGE_BREAK:                        # Seitengrenze als Marker behalten
            out.append(PAGE_BREAK); continue
        ln = ln.strip()
        if not ln or _TOC.search(ln): continue      # ToC-Punktlinien raus
        if _HDR.match(ln) or re.match(r'^\d{1,3}$', ln): continue  # Kopf-/Seitenzeilen
        if _FOOT.match(ln): continue                 # wiederkehrende Fusszeilen raus
        ln = re.sub(r'[ \t]{2,}', ' ', ln)           # Layout-Padding -> ein Leerzeichen
        out.append(ln)
    return out

def make_chunks(text: str, source: str, seed_tags=()):
    """seed_tags: optionale Tags aus z.B. .md-Frontmatter, die JEDEM Chunk dieses
    Dokuments mitgegeben werden (zusaetzlich zu den heuristisch erkannten)."""
    lines = _clean_lines(text)
    chunks = []; cur = []; head = ""; L = 0; ordn = 0
    seed = list(dict.fromkeys(t.lower() for t in seed_tags))   # dedupe, stabil
    def flush():
        nonlocal cur, L, ordn
        if cur:
            body = " ".join(cur).strip()
            if len(body) >= CHUNK_MIN:
                # Nur die Ueberschrift als Praefix (echte Struktur); KEIN Dateiname
                # mehr, der nur das Embedding-Fenster mit Rauschen fuellt.
                prefix = f"{head}: " if head else ""
                full = prefix + body
                # Tags aus Inhalt + Frontmatter-Seeds; Reihenfolge stabil, dedupe.
                tags = list(dict.fromkeys(_tag_chunk(full) + seed))
                chunks.append({"source": source, "ord": ordn, "text": full, "tags": tags})
                ordn += 1
        cur = []; L = 0
    for ln in lines:
        if ln == PAGE_BREAK:                         # Seiten-/Foliengrenze = harter Schnitt
            flush(); head = ""; continue             # Ueberschrift NICHT ueber Seiten schleppen
        m = _HEAD.match(ln)
        if m and re.search(r'[a-zäöü]', m.group(2)):  # echte Ueberschrift, kein Code-Fragment (R8 R2…)
            flush(); head = f"{m.group(1)} {m.group(2)}".strip()
            cur = [ln]; L = len(ln); continue
        mh = _MD_HEAD.match(ln)                       # Markdown-Ueberschrift ("## Titel")
        if mh:
            flush(); head = mh.group(2).strip()
            cur = [head]; L = len(head); continue     # '#'-Zeichen nicht in den Text uebernehmen
        if L + len(ln) > CHUNK_HARD: flush()
        cur.append(ln); L += len(ln) + 1
        if L >= CHUNK_TARGET and ln.endswith(('.', ';', ':')): flush()
    flush()
    return chunks

def _store_chunks(conn, chunks):
    pending = []; cur = conn.cursor()
    for c in chunks:
        cur.execute("INSERT INTO chunks(source, ord, text, vec, tags) VALUES(?,?,?,NULL,?)",
                    (c["source"], c["ord"], c["text"], json.dumps(c.get("tags") or [])))
        # Embedder bekommt NUR id+text — Tags bleiben absichtlich aussen vor.
        pending.append({"id": cur.lastrowid, "text": c["text"]})
    return pending

# --------------------------------------------------------------- Pure-Python BM25 (numpy-frei)
def _tok(s): return re.findall(r'[\wäöüÄÖÜ./-]+', s.lower())

class PureBM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.dl = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf = []; df = {}
        for d in corpus_tokens:
            f = {}
            for w in d: f[w] = f.get(w, 0) + 1
            self.tf.append(f)
            for w in f: df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    def scores(self, qtoks):
        out = [0.0] * self.N
        for i in range(self.N):
            f = self.tf[i]; dl = self.dl[i]; s = 0.0
            for w in qtoks:
                tf = f.get(w)
                if not tf: continue
                s += self.idf.get(w, 0.0) * (tf * (self.k1 + 1)) / \
                     (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out[i] = s
        return out

def rebuild_bm25():
    global BM25
    BM25 = PureBM25([it["toks"] for it in VECTORS]) if VECTORS else None

# --------------------------------------------------------------- Strukturierte Suchsyntax
# Schält Operatoren aus dem Fragetext: tags:frist | source:xyz | filename:*.pdf |
# "exakte phrase" | /regex/ | NOT wort | -wort. Der Rest bleibt Freitext und geht
# unveraendert ins Hybrid-Retrieval. Reiner Freitext (kein Operator) loest KEINEN
# Hard-Filter aus -> normale Fragen verhalten sich exakt wie bisher.
import fnmatch as _fnmatch

_RX_Q_FIELD  = re.compile(r'(\w+):("(?:[^"]*)"|\S+)')
_RX_Q_PHRASE = re.compile(r'"([^"]+)"')
_RX_Q_REGEX  = re.compile(r'/((?:[^/\\]|\\.)+)/')
_RX_Q_NOT    = re.compile(r'(?:^|\s)(?:NOT\s+|-)([^\s"]+)')

def parse_query(q):
    q = q or ""
    out = {"tags": [], "sources": [], "filenames": [], "phrases": [],
           "regexes": [], "excludes": [], "free": ""}
    def _grab_regex(m):
        out["regexes"].append(m.group(1)); return " "
    q = _RX_Q_REGEX.sub(_grab_regex, q)
    def _grab_field(m):
        key = m.group(1).lower(); val = m.group(2)
        if val.startswith('"') and val.endswith('"'): val = val[1:-1]
        val_l = val.lower()
        if key in ("tag", "tags"):              out["tags"].append(val_l)
        elif key in ("source", "src", "dok", "doc"): out["sources"].append(val_l)
        elif key in ("filename", "file", "datei"):   out["filenames"].append(val_l)
        else: return m.group(0)   # unbekanntes feld:wert -> als Freitext behalten
        return " "
    q = _RX_Q_FIELD.sub(_grab_field, q)
    def _grab_not(m):
        out["excludes"].append(m.group(1).lower()); return " "
    q = _RX_Q_NOT.sub(_grab_not, q)
    def _grab_phrase(m):
        out["phrases"].append(m.group(1)); return " "
    q = _RX_Q_PHRASE.sub(_grab_phrase, q)
    out["free"] = re.sub(r'\s+', ' ', q).strip()
    return out

def candidate_indices(parsed):
    """Indizes in VECTORS, die ALLE Hard-Constraints erfuellen. None = kein
    Constraint gesetzt (Aufrufer nutzt dann alle Chunks)."""
    if not any([parsed["tags"], parsed["sources"], parsed["filenames"],
                parsed["phrases"], parsed["regexes"], parsed["excludes"]]):
        return None
    compiled_rx = []
    for rx in parsed["regexes"]:
        try: compiled_rx.append(re.compile(rx, re.IGNORECASE))
        except re.error: pass
    keep = set()
    for idx, it in enumerate(VECTORS):
        text   = it.get("text") or ""
        text_l = text.lower()
        src_l  = (it.get("source") or "").lower()
        itags  = set(it.get("tags") or [])
        if parsed["tags"] and not all(t in itags for t in parsed["tags"]): continue
        if parsed["sources"] and not any(s in src_l for s in parsed["sources"]): continue
        if parsed["filenames"] and not any(_fnmatch.fnmatch(src_l, f) for f in parsed["filenames"]): continue
        if parsed["phrases"] and not all(p.lower() in text_l for p in parsed["phrases"]): continue
        if compiled_rx and not all(rx.search(text) for rx in compiled_rx): continue
        if parsed["excludes"]:
            toks = set(_tok(text))
            if any(x in toks or x in src_l for x in parsed["excludes"]): continue
        keep.add(idx)
    return keep

# --------------------------------------------------------------- Suche
def l2_normalize(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]

def _cosine_ranking(query_vec):
    q = l2_normalize(query_vec)
    scored = [(sum(a * b for a, b in zip(q, it["vec"])), idx) for idx, it in enumerate(VECTORS)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scored]

def _bm25_ranking(query_text, allowed=None):
    if not BM25 or not query_text: return []
    sc = BM25.scores(_tok(query_text))
    order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
    if allowed is not None:
        order = [i for i in order if i in allowed]
    return order

def hybrid_search(query_vec, query_text, k=TOP_K):
    """Reciprocal Rank Fusion ueber Cosine- und BM25-Ranking, plus optionalem,
    weichem Tag-Boost. Strukturierte Operatoren (tags:, source:, filename:,
    "phrase", /regex/, NOT/-) werden als HARTER Vorfilter angewandt; der Rest
    des Fragetextes ('free') treibt das semantische + lexikalische Ranking."""
    if not VECTORS: return []
    parsed  = parse_query(query_text or "")
    allowed = candidate_indices(parsed)        # None = keine Einschraenkung
    free    = parsed["free"] or (query_text or "")  # ganz ohne Freitext: Originalfrage nutzen

    rrf = {}
    # Vektor nur nutzen, wenn echter Freitext vorhanden ist. Bei reinen
    # Operator-Abfragen ('tags:frist') waere der Query-Vektor aus Syntax gebildet
    # und semantisch wertlos -> dann allein ueber Hard-Filter + BM25 gehen.
    use_vec = bool(query_vec) and bool(parsed["free"])
    if use_vec:
        for rank, idx in enumerate(_cosine_ranking(query_vec)):
            if allowed is not None and idx not in allowed: continue
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(_bm25_ranking(free, allowed)):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)

    # Sonderfall: NUR Hard-Filter, kein Freitext und kein Vektor-Treffer
    # (z.B. reine "tags:frist"-Abfrage) -> gefilterte Chunks unranked zurueckgeben.
    if not rrf and allowed:
        for idx in list(allowed)[:k]:
            rrf[idx] = 1.0
    if not rrf:
        return []

    if TAG_BOOST != 1.0:
        qtags = set(_tag_chunk(free))
        if qtags:
            for idx in rrf:
                if qtags & set(VECTORS[idx].get("tags") or []):
                    rrf[idx] *= TAG_BOOST
    top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(score, VECTORS[idx]) for idx, score in top]

# --------------------------------------------------------------- Routen
@app.route("/favicon.ico")
def favicon(): return send_from_directory(APP_DIR, "favicon.ico")

@app.route("/")
def index(): return send_from_directory(APP_DIR, "index.html")

@app.route("/static/<path:fname>")
def static_files(fname): return send_from_directory(STATIC_DIR, fname)

@app.route("/models/<path:fname>")
def model_files(fname):
    target = (MODELS_DIR / fname).resolve()
    if MODELS_DIR.resolve() not in target.parents and target != MODELS_DIR.resolve():
        abort(403)
    return send_from_directory(MODELS_DIR, fname)

@app.route("/api/status")
def status():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    emb   = conn.execute("SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL").fetchone()[0]
    docs  = conn.execute("SELECT COUNT(DISTINCT source) FROM chunks").fetchone()[0]
    return jsonify({"chunks": total, "embedded": emb, "docs": docs, "pdf_ok": PDF_OK})

@app.route("/api/models")
def models():
    # Embedding-Modelle gehoeren NICHT ins LLM-Dropdown. Statt einer
    # Einzelausnahme ("MiniLM") generisch alle bekannten Embedder-Namensmuster
    # ausschliessen — sonst rutscht z.B. multilingual-e5-large-instruct durch
    # und kann versehentlich als Text-Generator gewaehlt werden (-> Kauderwelsch).
    EMBED_MARKERS = ("minilm", "e5-", "-e5", "embed", "bge", "gte-", "nomic", "sentence")
    found = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if not d.is_dir():
                continue
            low = d.name.lower()
            if any(m in low for m in EMBED_MARKERS):
                continue
            found.append(d.name)
    return jsonify({"llm": found})

@app.route("/api/reset", methods=["POST"])
def reset():
    conn = get_db(); conn.execute("DELETE FROM chunks"); conn.commit()
    VECTORS.clear(); rebuild_bm25()
    return jsonify({"ok": True})

@app.route("/api/documents")
def documents():
    conn = get_db()
    rows = conn.execute(
        "SELECT source AS name, COUNT(*) AS chunks, "
        "SUM(CASE WHEN vec IS NOT NULL THEN 1 ELSE 0 END) AS embedded "
        "FROM chunks GROUP BY source ORDER BY source COLLATE NOCASE").fetchall()
    return jsonify({"documents": [dict(r) for r in rows]})

@app.route("/api/delete_doc", methods=["POST"])
def delete_doc():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Kein Dokumentname angegeben."}), 400
    conn = get_db()
    conn.execute("DELETE FROM chunks WHERE source=?", (name,)); conn.commit()
    VECTORS[:] = [it for it in VECTORS if it["source"] != name]   # In-Memory-Index angleichen
    rebuild_bm25()
    return jsonify({"ok": True, "deleted": name})

def _ingest_one(conn, name, src, report, pending):
    try: text, seed_tags = extract_text_any(name, src)
    except Exception as e: report.append({"file": name, "status": f"Lesefehler: {e}"}); return
    if not text or not text.strip():
        report.append({"file": name, "status": "LEER (kein Textlayer?)"}); return
    chunks = make_chunks(text, name, seed_tags)
    pending.extend(_store_chunks(conn, chunks))
    report.append({"file": name, "status": f"OK, {len(chunks)} Chunks"})

def _is_temp(name: str) -> bool:
    # Office-Sperrdateien (~$...) und versteckte Dateien ueberspringen
    return name.startswith("~$") or name.startswith(".")

@app.route("/api/ingest", methods=["POST"])
def ingest():
    files = request.files.getlist("files")
    if not files: return jsonify({"error": "Keine Dateien empfangen."}), 400
    conn = get_db(); report, pending = [], []
    for f in files:
        if _is_temp(f.filename or ""):
            continue
        _ingest_one(conn, f.filename, io.BytesIO(f.read()), report, pending)
    conn.commit()
    return jsonify({"report": report, "pending": pending})

@app.route("/api/ingest_folder", methods=["POST"])
def ingest_folder():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = get_db()
        existing = {r["source"] for r in conn.execute("SELECT DISTINCT source FROM chunks")}
        report, pending = [], []
        found = sorted((p for p in DATA_DIR.rglob("*")
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT and not _is_temp(p.name)),
                       key=lambda p: p.name.lower())
        if not found:
            return jsonify({"report": [{"file": str(DATA_DIR),
                                        "status": "Keine PDF/DOCX/XLSX gefunden"}], "pending": []})
        for doc in found:
            if doc.name in existing:
                report.append({"file": doc.name, "status": "bereits indiziert – übersprungen"}); continue
            _ingest_one(conn, doc.name, str(doc), report, pending)
        conn.commit()
        return jsonify({"report": report, "pending": pending})
    except Exception as e:
        log.exception("ingest_folder")
        return jsonify({"error": str(e), "report": [], "pending": []}), 500

@app.route("/api/store_vectors", methods=["POST"])
def store_vectors():
    data = request.get_json(force=True, silent=True) or {}
    conn = get_db(); n = 0
    for it in data.get("items", []):
        cid, vec = it.get("id"), it.get("vector")
        if cid is None or not vec: continue
        nv = l2_normalize(vec)
        conn.execute("UPDATE chunks SET vec=? WHERE id=?", (json.dumps(nv), cid))
        row = conn.execute("SELECT source, text, tags FROM chunks WHERE id=?", (cid,)).fetchone()
        if row:
            VECTORS.append({"id": cid, "source": row["source"], "text": row["text"],
                            "vec": nv, "toks": _tok(row["text"]),
                            "tags": json.loads(row["tags"]) if row["tags"] else []})
            n += 1
    conn.commit()
    rebuild_bm25()                # BM25-Index nach Vektor-Update neu aufbauen
    return jsonify({"stored": n})

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(force=True, silent=True) or {}
    qvec = data.get("vector")
    qtext = data.get("query") or data.get("text") or ""   # NEU: Frontend sendet auch den Fragetext
    if not qvec and not qtext:
        return jsonify({"error": "Weder Vektor noch Text."}), 400
    if not VECTORS:
        return jsonify({"hits": []})
    top = hybrid_search(qvec, qtext, TOP_K)
    return jsonify({"hits": [{"source": it["source"], "text": it["text"], "score": round(s, 4)}
                             for s, it in top]})

# ----------------------------------------------------------------
if __name__ == "__main__":
    import traceback, socket
    LOGFILE = APP_DIR / "armachat_error.log"

    def _fatal(msg):
        try:
            with open(LOGFILE, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n" + "-" * 60 + "\n")
        except Exception:
            pass
        print("\n" + msg)
        try:
            input("\n[Enter] zum Schliessen — Fehler steht auch in armachat_error.log")
        except Exception:
            pass

    # 1) Verzeichnisse + DB
    try:
        STATIC_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True); DATA_DIR.mkdir(exist_ok=True)
        init_db()
    except Exception:
        _fatal("FEHLER beim Start (Verzeichnisse/DB):\n" + traceback.format_exc())
        raise SystemExit(1)

    # 2) Port pruefen, damit "belegt" lesbar gemeldet wird statt Stacktrace-Blitz
    _probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _probe.bind((HOST, PORT)); _probe.close()
    except OSError:
        _probe.close()
        _fatal(f"Port {PORT} ist belegt — armachat laeuft vermutlich schon.\n"
               f"  -> Im Browser oeffnen:  http://localhost:{PORT}\n"
               f"  -> Oder alte python.exe im Task-Manager beenden und neu starten.")
        raise SystemExit(1)

    # 3) Browser-Autostart: unter AppLocker (Python-initiierter Start) ggf. blockiert -> abfangen
    import threading, time
    def _browser():
        time.sleep(2.5)
        try:
            import webbrowser; webbrowser.open(f"http://localhost:{PORT}")
        except Exception:
            pass  # nicht startrelevant; Seite manuell oeffnen
    threading.Thread(target=_browser, daemon=True).start()

    log.info("armachat laeuft: http://localhost:%d  (zum Beenden dieses Fenster schliessen)", PORT)
    try:
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except Exception:
        _fatal("FEHLER beim Serverstart:\n" + traceback.format_exc())
        raise SystemExit(1)
