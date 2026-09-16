# -*- coding: utf-8 -*-
"""
tiseR - RAG-Backend   [Version 20260916v02] (AppLocker-konform, KEINE nativen DLLs)
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
  8. GarbageCheck am INGEST (20260815v01): zeichenstatistische Erkennung kaputter
     Textlayer (Browser-PDF-Exporte mit subgesetzten Fonts). Laeuft per Default im
     MESSMODUS (GARBAGE_REJECT = False) — misst und meldet, verwirft nichts.
  9. Eval-Harness Variante B (20260904v01): optionale Einbindung von eval_module.py
     am Ende der Routen-Sektion. Misst Recall@k / Coverage / MRR des ECHTEN
     Hybrid-Retrievals gegen ein XLSX-Frage/Antwort-Set. Fehlt das Modul, laeuft
     tiseR unveraendert weiter.
 10. Glossar-Modus fuer XLSX unter "eval" (20260905v01): Ortsregel statt Formatregel.
     Eine XLSX unterhalb von "eval" wird als Glossar (Frage|Antwort|Thema|Quelle) gelesen
     und je Zeile zu einem eigenen Chunk mit '## Begriff'-Ueberschrift. Blaetter mit
     '_'-Praefix bleiben reine Messdaten und gehen NIE in den Index. Jede XLSX ausserhalb
     von "eval" laeuft unveraendert durch extract_xlsx_text(). Der Ingest ist fuer diese
     Dateien idempotent (alte Chunks werden ersetzt statt dupliziert).
 11. Glossar-Upload (20260905v02): /api/ingest_glossary nimmt eine XLSX entgegen,
     legt sie in "eval" ab und indiziert sie im Glossar-Modus. Eigener Knopf im
     Dokumente-Dialog, getrennt vom normalen Upload.
 12. Anonymisierung (20260908v01): optionale Einbindung von anon_module.py am Ende
     der Routen-Sektion. Markiert schuetzenswerte Stellen einer hochgeladenen Datei
     (Strukturmuster aus anon_patterns.json + Begriffe aus dem Blatt '_anon' der
     Glossar-XLSX) und ersetzt sie erst nach manueller Freigabe im Browser.
     ENTSCHEIDET NICHT ueber Klassifizierung — ein anonymisiertes Dokument bleibt
     eingestuft. Fehlt das Modul, laeuft tiseR unveraendert weiter.
 13. Dubletten-Check + Kontext-Vorspann (20260916v01):
     - Jeder Chunk traegt den SHA-1 des extrahierten Dokumenttexts (Spalte doc_hash).
       Gleicher Inhalt (egal welcher Name) -> uebersprungen; gleicher Name, neuer
       Inhalt -> alte Chunks ersetzt. Schuetzt df/avgdl im BM25 vor Mehrfach-"Laden".
     - CTX_PREFIX (Default False): "Dokumenttitel > Ueberschrift: " vor jedem Chunk
       (Embedding UND BM25). Wirkt nur auf NEU eingelesene Dokumente -> fuer die
       Messung Korpus zuruecksetzen und neu einlesen.
 14. Vektoren als float32-BLOB + DB-Name tiser.db (20260916v02):
     - vec wird als float32-BLOB (array-Modul, numpy-frei) statt JSON gespeichert.
       Spart ca. 75-85 % DB-Groesse und json.loads beim Start.
     - Beim 1. Start: armachat.db wird nach tiser.db KOPIERT (Original bleibt als
       Rollback-Sicherung liegen), JSON-Vektoren werden konvertiert, danach VACUUM.
     - Dimensionspruefung beim Laden: Vektoren mit abweichender Dimension werden
       NICHT geladen und im Log gemeldet (Schutz bei Embedding-Modellwechsel).
     - ACHTUNG: Aeltere tiseR-Versionen koennen tiser.db nicht lesen.
PDF-Extraktion: pypdf (reines Python).
"""
import io, json, math, re, sqlite3, logging, zipfile, hashlib, shutil, sys
from array import array
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, g, abort

try:
    from pypdf import PdfReader
    PDF_OK = True
except Exception as _e:
    PDF_OK, _PDF_ERR = False, str(_e)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tiser")

APP_DIR    = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODELS_DIR = APP_DIR / "models"
DATA_DIR   = APP_DIR / "data"
EVAL_DIR   = APP_DIR / "eval"   # Glossar-/Eval-XLSX (Sonderbehandlung, siehe unten)
DB_PATH    = APP_DIR / "tiser.db"
LEGACY_DB  = APP_DIR / "armachat.db"   # 20260916v02: nur fuer einmalige Migration
HOST, PORT = "127.0.0.1", 8000

CHUNK_TARGET = 500      # Ziel-Zeichen/Chunk: passt ins Embedding-Fenster (~128 Tokens)
CHUNK_HARD   = 750      # harte Obergrenze
CHUNK_MIN    = 15       # Mindestlaenge (sonst gehen kurze Folien beim Seiten-Flush verloren)
TOP_K        = 6      # 20260805v04: von 4 auf 6. Der Worker schnitt zusaetzlich auf
                        # slice(0,3) -> effektiv sahen die Modelle 3 Chunks. Bei
                        # 500-Zeichen-Chunks aus Mail-Korpora ist das zu wenig Substanz.
RRF_K        = 60       # RRF-Konstante (Standardwert)
PAGE_BREAK   = "\x0c"   # Seiten-/Foliengrenze -> harter Chunk-Schnitt
# Metadaten-Boost: weicher, MULTIPLIKATIVER Faktor auf Chunks, deren Tags zum
# Fragetyp passen. 1.0 = AUS (Feature verdrahtet, aber wirkungslos). Erst auf
# z.B. 1.15 erhoehen, NACHDEM eval_harness.py Hybrid+Boost gegen reines BM25
# gemessen hat. Vorher waere jede Zahl reines Bauchgefuehl.
TAG_BOOST    = 1.0
# 20260916v01: Dokumenttitel als Chunk-Vorspann. Aus bis Golden-Set-Messung den Nutzen belegt.
CTX_PREFIX   = False
CTX_TITLE_MAX = 60

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

# --------------------------------------------------------------- Vektor-Serialisierung (20260916v02)
def vec_to_blob(vec) -> bytes:
    a = array("f", vec)
    if sys.byteorder != "little": a.byteswap()   # auf Platte immer little-endian
    return a.tobytes()

def blob_to_vec(raw) -> list:
    if isinstance(raw, str):                      # Altbestand (JSON), falls Migration unvollstaendig
        return json.loads(raw)
    a = array("f"); a.frombytes(raw)
    if sys.byteorder != "little": a.byteswap()
    return a.tolist()

def _migrate_legacy_db():
    """armachat.db -> tiser.db kopieren (nicht verschieben: Rollback bleibt moeglich)."""
    if not DB_PATH.exists() and LEGACY_DB.exists():
        shutil.copy2(LEGACY_DB, DB_PATH)
        log.info("Migration: %s nach %s kopiert (Original bleibt als Sicherung).",
                 LEGACY_DB.name, DB_PATH.name)

def _migrate_vec_blobs(conn):
    """JSON-Vektoren einmalig in float32-BLOBs umwandeln, danach VACUUM."""
    rows = conn.execute("SELECT id, vec FROM chunks WHERE typeof(vec)='text'").fetchall()
    if not rows: return
    for r in rows:
        conn.execute("UPDATE chunks SET vec=? WHERE id=?", (vec_to_blob(json.loads(r["vec"])), r["id"]))
    conn.commit()
    conn.execute("VACUUM")
    log.info("Migration: %d Vektoren von JSON auf float32-BLOB umgestellt.", len(rows))

def init_db():
    _migrate_legacy_db()
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
    try:   # 20260916v01: Dokument-Hash fuer Dubletten-Check
        conn.execute("ALTER TABLE chunks ADD COLUMN doc_hash TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS ix_chunks_hash ON chunks(doc_hash)")
    conn.commit()
    _migrate_vec_blobs(conn)
    VECTORS.clear()
    dim = None; bad = 0
    for r in conn.execute("SELECT id, source, text, vec, tags FROM chunks WHERE vec IS NOT NULL"):
        v = blob_to_vec(r["vec"])
        if dim is None: dim = len(v)
        if len(v) != dim:          # Mischbestand aus zwei Embedding-Modellen -> nicht laden
            bad += 1; continue
        VECTORS.append({"id": r["id"], "source": r["source"], "text": r["text"],
                        "vec": v, "toks": _tok(r["text"]),
                        "tags": json.loads(r["tags"]) if r["tags"] else []})
    conn.close()
    rebuild_bm25()
    if bad:
        log.warning("%d Chunks mit abweichender Vektor-Dimension (erwartet %d) NICHT geladen "
                    "-> Korpus zuruecksetzen und neu einlesen.", bad, dim)
    log.info("DB bereit. %d Chunks mit Vektor (Dim %s).", len(VECTORS), dim)

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

# --- MIME-encoded-words (RFC 2047) ----------------------------------------
# In WEITERGELEITETEN Mails stehen die Header des Originals als TEXT im Body:
#   "Von: =?utf-8?B?QnJlY2hiw7xobCBGYWJpYW4gQVJNQVNVSVNTRQ==?= <...>"
# policy.default dekodiert nur die ECHTEN Header, nicht diese Textzeilen. Der
# Base64-Salat landete also in Chunk UND Embedding: er frisst Kontextfenster,
# zerstoert die Namens-Semantik und tauchte in den Testantworten woertlich auf.
# Hier wird jedes encoded-word im gesamten extrahierten Text aufgeloest.
_RX_MIME_WORD = re.compile(r'=\?([\w\-]+)\?([BbQq])\?([^?]*)\?=')

def _decode_mime_words(text: str) -> str:
    if not text or "=?" not in text:
        return text
    import base64, quopri
    def _one(m):
        charset, enc, payload = m.group(1), m.group(2).upper(), m.group(3)
        try:
            if enc == "B":
                raw = base64.b64decode(payload + "=" * (-len(payload) % 4))
            else:
                raw = quopri.decodestring(payload.replace("_", " "))
            return raw.decode(charset, "replace")
        except Exception:
            return m.group(0)          # unlesbar -> unveraendert stehen lassen
    return _RX_MIME_WORD.sub(_one, text)

def _sender_label(raw: str) -> str:
    """Lesbarer Absendername aus einem From-Header.
    'Brechbühl Fabian <f.b@vbs.admin.ch>' -> 'Brechbühl Fabian'.
    Fallback: lokaler Teil der Adresse."""
    if not raw: return ""
    raw = _decode_mime_words(raw).strip()
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
    # Ganzer Text durch den MIME-Dekoder: erwischt auch die als Body-Text
    # eingebetteten Header weitergeleiteter Mails.
    return _decode_mime_words("\n".join(md_head + head + ["", body.strip()]))

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
        return _decode_mime_words("\n".join(md_head + head + ["", body.strip()]))
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

# --------------------------------------------------------------- Glossar-XLSX (nur eval\)
# WARUM EIN SONDERWEG: extract_xlsx_text() macht aus jeder Zeile "Zelle | Zelle | …".
# Fuer Datentabellen ist das richtig. Fuer ein Glossar ist es falsch: die Struktur
# "Begriff -> Definition" geht verloren, make_chunks() findet keine Ueberschrift und
# setzt deshalb KEINEN Praefix, und aufeinanderfolgende Eintraege laufen ineinander
# (ein Chunk endet mitten im naechsten Begriff). Gemessen an der realen Datei:
# 380 praefixlose Chunks statt 63 sauber geschnittener.
#
# ORTSREGEL statt Formatregel: XLSX-Dateien UNTERHALB von eval\ werden als Glossar
# gelesen (Kopfzeile Frage | Antwort | Thema | Quelle). Jede andere XLSX-Datei laeuft
# unveraendert durch extract_xlsx_text(). Damit aendert sich am bestehenden Verhalten
# fuer alle bisherigen Dokumente NICHTS.
#
# Erzeugt wird Markdown: "## <Begriff>" + Frage + Antwort. Die Frage bleibt im Text,
# weil sie die natuerlichsprachliche Formulierung liefert, nach der Nutzer suchen —
# der Begriff allein deckt das Vokabular nicht ab.
GLOSSARY_SHEET_SKIP = "_"   # Blaetter mit diesem Praefix sind REINE Messdaten (nie Index)

def _xlsx_rows(src):
    """Liefert (blattname, [zellwerte]) je Zeile. Spaltenposition bleibt erhalten,
    auch wenn leere Zellen in der XML fehlen."""
    def _col(ref):
        n = 0
        for ch in ref:
            if ch.isalpha(): n = n * 26 + (ord(ch.upper()) - 64)
            else: break
        return n - 1
    _RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            with z.open("xl/sharedStrings.xml") as f:
                for si in ET.parse(f).getroot().iter(_S + "si"):
                    shared.append("".join(t.text or "" for t in si.iter(_S + "t")))
        sheets = []
        try:
            with z.open("xl/workbook.xml") as f:
                wbroot = ET.parse(f).getroot()
            rels = {}
            with z.open("xl/_rels/workbook.xml.rels") as f:
                for rel in ET.parse(f).getroot():
                    rels[rel.get("Id")] = rel.get("Target")
            for sh in wbroot.iter(_S + "sheet"):
                tgt = (rels.get(sh.get(_RNS + "id"), "") or "").lstrip("/")
                if not tgt.startswith("xl/"): tgt = "xl/" + tgt
                sheets.append((sh.get("name"), tgt))
        except Exception:
            sheets = [(n.split("/")[-1], n) for n in sorted(names)
                      if n.startswith("xl/worksheets/") and n.endswith(".xml")]
        for sheet_name, target in sheets:
            if target not in names: continue
            with z.open(target) as f:
                wroot = ET.parse(f).getroot()
            for row in wroot.iter(_S + "row"):
                vals = {}
                for c in row.iter(_S + "c"):
                    typ = c.get("t"); ref = c.get("r") or ""; out = ""
                    if typ == "s":
                        v = c.find(_S + "v")
                        if v is not None and v.text is not None:
                            try: out = shared[int(v.text)]
                            except (ValueError, IndexError): out = ""
                    elif typ == "inlineStr":
                        isn = c.find(_S + "is")
                        if isn is not None:
                            out = "".join(x.text or "" for x in isn.iter(_S + "t"))
                    else:
                        v = c.find(_S + "v")
                        out = (v.text or "") if v is not None else ""
                    vals[_col(ref)] = (out or "").strip()
                if vals:
                    width = max(vals) + 1
                    yield sheet_name, [vals.get(i, "") for i in range(width)]

def _term_from(frage: str, thema: str) -> str:
    """Ueberschrift eines Glossar-Chunks. Bevorzugt Spalte 'Thema' (dort steht der
    Begriff). Fehlt sie, wird der Begriff aus der Frage geschaelt
    ('Was ist X (auch Y)?' -> 'X (auch Y)')."""
    if thema: return thema
    m = re.match(r'^\s*(?:Was\s+(?:ist|bedeutet|sind)|Wofuer\s+steht|Wofür\s+steht)\s+(.+?)\s*\??$',
                 frage or "", re.I)
    term = (m.group(1) if m else (frage or "")).strip(" ?")
    return term[:80]

def extract_xlsx_glossary_text(src) -> str:
    """XLSX mit Kopfzeile Frage|Antwort|Thema|Quelle -> Markdown-Glossar.
    Blaetter mit GLOSSARY_SHEET_SKIP-Praefix werden uebersprungen: sie sind
    Messdaten (z.B. Mailkorpus-Fragen) und duerfen NIE in den Index, sonst misst
    die Evaluation sich selbst."""
    out, cur_sheet, header_seen = [], None, False
    for sheet, vals in _xlsx_rows(src):
        if sheet != cur_sheet:
            cur_sheet, header_seen = sheet, False
        if str(sheet or "").startswith(GLOSSARY_SHEET_SKIP):
            continue
        get = lambda i: vals[i] if i < len(vals) else ""
        frage, antwort, thema, quelle = get(0), get(1), get(2), get(3)
        if not header_seen:
            header_seen = True
            if frage.lower().startswith(("frage", "begriff", "term")):
                continue                       # Kopfzeile
        if not frage or not antwort:
            continue                           # unvollstaendige Zeile still ueberspringen
        if out: out.append(PAGE_BREAK)         # harter Schnitt zwischen Eintraegen
        out.append("## " + _term_from(frage, thema))
        line = frage.rstrip("?") + "? " + antwort
        if quelle: line += f" (Quelle: {quelle}; Blatt: {sheet})"
        out.append(line)
    return "\n".join(out)

def _is_glossary_path(src) -> bool:
    """True, wenn der Pfad unterhalb von eval\\ liegt. Nur Pfade, keine BytesIO —
    ein Upload ueber 'Laden' bleibt bewusst eine normale XLSX."""
    if hasattr(src, "read"): return False
    try:
        return EVAL_DIR.resolve() in Path(src).resolve().parents
    except Exception:
        return False


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
    if ext == ".xlsx" and _is_glossary_path(src):
        # Ortsregel: XLSX unter eval\ ist ein Glossar. Seed-Tag 'glossar' macht
        # 'tags:glossar' als Suchoperator sofort nutzbar.
        return extract_xlsx_glossary_text(src), ["glossar"]
    if ext in (".md", ".markdown"):
        if hasattr(src, "read"):
            seed_tags = _md_seed_tags(src)
            src.seek(0)                      # Stream fuer den Handler zuruecksetzen
        else:
            seed_tags = _md_seed_tags(src)
    return handler(src), seed_tags

# --------------------------------------------------------------- GarbageCheck (Ingest-Qualitaetsfilter)
# ANLASS: Ein per Browser "Drucken -> PDF" erzeugtes Web-PDF hat oft einen Textlayer
# mit subgesetzten Fonts ohne verwertbares ToUnicode-CMap. pypdf liefert daraus
# Zeichensalat ("+,, -../ 0\t1+ 23456789:4"). Der floss bisher UNBEMERKT durch:
# _clean_lines() filtert ihn nicht, CHUNK_MIN=15 laesst ihn durch, der Ingest-Report
# meldet "OK, n Chunks". Der Schaden bleibt NICHT lokal:
#   - BM25: jedes Salat-Token ist ein Hapax -> maximales IDF. Kollidiert eines
#     zufaellig mit einem Fragetoken, springt der Muell-Chunk nach oben. Zugleich
#     verschiebt sich avgdl und damit die Laengennormalisierung ALLER Chunks.
#   - Embedding: e5-large bildet Salat nicht auf "nichts" ab, sondern auf einen
#     beliebigen Punkt — bei Fragen ohne guten echten Treffer eine plausible
#     Cosine-Nachbarschaft.
#   - RRF belohnt Konsens zweier Ranker und verstaerkt den Zufallstreffer.
# Symptom fuer den Nutzer: "Dazu enthalten die Dokumente keine Angabe", obwohl die
# Antwort indiziert ist. eval_harness.py sieht das NICHT (misst nur BM25-Recall).
#
# SCHARFSCHALTUNG: GARBAGE_REJECT bleibt False (Messmodus). Es wird NICHTS verworfen,
# nur gemessen und im Ingest-Report ausgewiesen. Erst wenn die Werte gegen den
# ECHTEN Korpus gemessen sind (Tabellen aus XLSX und abkuerzungslastige Chunks
# liegen nahe an denselben Schwellen), auf True setzen — gleiche Disziplin wie
# bei TAG_BOOST.
GARBAGE_REJECT   = False   # True = Muell-Chunks verwerfen / Dokument ablehnen
GARBAGE_DOC_MAX  = 0.30    # ab diesem Anteil Muell-Chunks gilt das DOKUMENT als kaputt

# KALIBRIERUNG 20260815: Der erste Entwurf mass "Zeichen ausserhalb des erwarteten
# Alphabets" und schlug am Referenzfall NICHT an — pypdf liefert bei subgesetzten
# Fonts keinen Mojibake, sondern Glyph-Indizes ("/0 /1 /2 /3"), also reines ASCII.
# Gemessen wird daher Buchstabenanteil und Wortdichte.
_RX_MOJIBAKE = re.compile(r"[\uFFFD\u0080-\u009F]")
_RX_ALPHA    = re.compile(r"[^\W\d_]", re.UNICODE)
_RX_WORD     = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# Schwellen (siehe Kalibrierung im Kopf dieses Abschnitts)
GARBAGE_ALPHA    = 0.35    # Buchstabenanteil; deutscher Fliesstext liegt bei ~0.75
GARBAGE_WORDDENS = 3.0     # Woerter (>=3 Buchstaben) je 100 Zeichen; Fliesstext ~12

def _garbage_metrics(text: str):
    """Liefert (alpha_ratio, word_density, mojibake_ratio).
    Rein zeichenstatistisch, kein Sprachmodell — billig und deterministisch."""
    n = len(text or "")
    if not n:
        return 1.0, 100.0, 0.0
    alpha = len(_RX_ALPHA.findall(text)) / n
    dens  = len(_RX_WORD.findall(text)) * 100.0 / n
    moji  = len(_RX_MOJIBAKE.findall(text)) / n
    return alpha, dens, moji

def _is_garbage(text: str) -> bool:
    """Zwei unabhaengige Signaturen kaputter Extraktion:
      (A) Glyph-Index-Fallback: pypdf gibt '/0 /1 /2 …' aus, wenn der Font
          subgesetzt ist und keine ToUnicode-CMap mitbringt. Ergebnis: fast keine
          Buchstaben, gar keine Woerter. Beide Kriterien MUESSEN reissen — reine
          Zahlenkolonnen haben zwar wenig Buchstaben, aber auch wenig Zeichen
          insgesamt und werden ueber die Wortdichte nicht mitgerissen.
      (B) Mojibake: Ersetzungszeichen und C1-Steuerzeichen aus falsch geratener
          Kodierung."""
    alpha, dens, moji = _garbage_metrics(text)
    return (alpha < GARBAGE_ALPHA and dens < GARBAGE_WORDDENS) or moji > 0.02

def screen_chunks(chunks, source: str):
    """Bewertet die Chunks EINES Dokuments.
    Rueckgabe (kept, stats). stats = {n, bad, ratio, doc_bad, badchar_avg, novowel_avg}.
    Bei GARBAGE_REJECT=False ist kept IMMER == chunks (reiner Messmodus)."""
    n = len(chunks)
    stats = {"n": n, "bad": 0, "ratio": 0.0, "doc_bad": False,
             "alpha_avg": 1.0, "dens_avg": 100.0}
    if not n:
        return chunks, stats
    flags, asum, dsum = [], 0.0, 0.0
    for c in chunks:
        a, d, _mo = _garbage_metrics(c["text"])
        asum += a; dsum += d
        flags.append(_is_garbage(c["text"]))
    stats["bad"]       = sum(flags)
    stats["ratio"]     = stats["bad"] / n
    stats["alpha_avg"] = asum / n
    stats["dens_avg"]  = dsum / n
    stats["doc_bad"]     = stats["ratio"] > GARBAGE_DOC_MAX
    log.info("GarbageCheck %s: %d/%d Chunks auffaellig (%.0f%%), "
             "alpha_avg=%.3f dens_avg=%.1f%s",
             source, stats["bad"], n, stats["ratio"] * 100,
             stats["alpha_avg"], stats["dens_avg"],
             "  -> DOKUMENT waere abgelehnt" if stats["doc_bad"] else "")
    if not GARBAGE_REJECT:
        return chunks, stats                      # Messmodus: nichts verwerfen
    if stats["doc_bad"]:
        return [], stats                          # ganzes Dokument ablehnen
    return [c for c, f in zip(chunks, flags) if not f], stats

def _garbage_note(stats) -> str:
    """Kurzer Zusatz fuer den Ingest-Report. Immer sichtbar, damit die Schwellen
    am echten Korpus kalibriert werden koennen, bevor irgendetwas verworfen wird."""
    if not stats["n"] or not stats["bad"]:
        return ""
    return (f" · GarbageCheck: {stats['bad']}/{stats['n']} auffaellig "
            f"({stats['ratio']*100:.0f}%)")

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

_RX_TITLE_ID = re.compile(r'(?<![A-Za-z0-9])(?:ar-)?[A-Z]-?[0-9A-F]{6,}(?:/\d+)?(?![A-Za-z0-9])|(?<![A-Za-z0-9])v?\d{8}v\d{2}(?![A-Za-z0-9])')

def _doc_title(source: str) -> str:
    """Lesbarer Titel aus dem Dateinamen: Endung, Dok-IDs, Versionsstempel und
    Trennzeichen raus. Leer, wenn nichts Sinnvolles uebrig bleibt."""
    t = _RX_TITLE_ID.sub(" ", Path(source).stem)
    t = re.sub(r'\s+', ' ', re.sub(r'[_\-.]+', ' ', t)).strip()
    if len(re.findall(r'[A-Za-zÄÖÜäöü]', t)) < 3: return ""
    return t[:CTX_TITLE_MAX].rstrip()

def make_chunks(text: str, source: str, seed_tags=()):
    """seed_tags: optionale Tags aus z.B. .md-Frontmatter, die JEDEM Chunk dieses
    Dokuments mitgegeben werden (zusaetzlich zu den heuristisch erkannten)."""
    lines = _clean_lines(text)
    chunks = []; cur = []; head = ""; L = 0; ordn = 0
    seed = list(dict.fromkeys(t.lower() for t in seed_tags))   # dedupe, stabil
    title = _doc_title(source) if CTX_PREFIX else ""
    def flush():
        nonlocal cur, L, ordn
        if cur:
            body = " ".join(cur).strip()
            if len(body) >= CHUNK_MIN:
                # Nur die Ueberschrift als Praefix (echte Struktur); KEIN Dateiname
                # mehr, der nur das Embedding-Fenster mit Rauschen fuellt.
                parts = [p for p in (title, head) if p]   # 20260916v01: CTX_PREFIX
                prefix = (" > ".join(parts) + ": ") if parts else ""
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

def _store_chunks(conn, chunks, doc_hash=None):
    pending = []; cur = conn.cursor()
    for c in chunks:
        cur.execute("INSERT INTO chunks(source, ord, text, vec, tags, doc_hash) VALUES(?,?,?,NULL,?,?)",
                    (c["source"], c["ord"], c["text"], json.dumps(c.get("tags") or []), doc_hash))
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
    # --- 20260916v01: Dubletten-Check ueber Inhalts-Hash (extrahierter Text) ----
    h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    dup = conn.execute("SELECT source FROM chunks WHERE doc_hash=? LIMIT 1", (h,)).fetchone()
    if dup:
        report.append({"file": name, "status": "unverändert – übersprungen" if dup["source"] == name
                       else f"Dublette von '{dup['source']}' – übersprungen"}); return
    replaced = conn.execute("SELECT COUNT(*) FROM chunks WHERE source=?", (name,)).fetchone()[0]
    if replaced:   # gleicher Name, neuer (oder nie gehashter) Inhalt -> ersetzen
        conn.execute("DELETE FROM chunks WHERE source=?", (name,))
        VECTORS[:] = [it for it in VECTORS if it["source"] != name]
        rebuild_bm25()
    chunks = make_chunks(text, name, seed_tags)
    kept, gstats = screen_chunks(chunks, name)
    if GARBAGE_REJECT and gstats["doc_bad"]:
        report.append({"file": name, "status":
            f"ABGELEHNT: Textlayer unbrauchbar ({gstats['ratio']*100:.0f}% Zeichensalat). "
            f"Vermutlich Browser-'Drucken -> PDF' ohne einbettbare Fonts. "
            f"Original-Datei statt PDF-Export verwenden."})
        return
    pending.extend(_store_chunks(conn, kept, h))
    dropped = len(chunks) - len(kept)
    status  = f"OK, {len(kept)} Chunks"
    if replaced: status += f" (Version ersetzt, {replaced} alte Chunks entfernt)"
    if dropped: status += f" ({dropped} als Zeichensalat verworfen)"
    report.append({"file": name, "status": status + _garbage_note(gstats)})

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

@app.route("/api/ingest_glossary", methods=["POST"])
def ingest_glossary():
    """Glossar-Upload: speichert die XLSX nach eval\\ und indiziert sie im
    Glossar-Modus (siehe extract_xlsx_glossary_text).

    ZWEI ROLLEN, EINE DATEI — bewusst so:
      eval\\  ist Ablageort UND Messquelle. Der Harness liest dieselbe Datei.
      Blaetter mit '_'-Praefix gehen NIE in den Index; sie bleiben reine Messdaten.

    IDEMPOTENT: vorhandene Chunks derselben Datei werden ersetzt, nicht ergaenzt.
    Ein Glossar ist ein gepflegtes Dokument — wiederholtes Hochladen nach einer
    Ergaenzung muss funktionieren, ohne df/avgdl im BM25 zu verfaelschen."""
    ups = request.files.getlist("files")
    if not ups:
        return jsonify({"error": "Keine Datei erhalten."}), 400
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    report, pending = [], []
    conn = get_db()
    try:
        for up in ups:
            name = Path(up.filename or "").name
            if not name:
                continue
            if not name.lower().endswith(".xlsx"):
                report.append({"file": name,
                               "status": "übersprungen – Glossar muss .xlsx sein "
                                         "(Frage | Antwort | Thema | Quelle)"})
                continue
            dest = EVAL_DIR / name
            up.save(str(dest))

            # Vorabpruefung: liefert die Datei ueberhaupt Glossarzeilen? Eine XLSX
            # ohne passende Kopfzeile ergibt leeren Text — das soll der Nutzer
            # erfahren, statt still 0 Chunks zu bekommen.
            try:
                probe = extract_xlsx_glossary_text(str(dest))
            except Exception as e:
                report.append({"file": name, "status": f"Fehler beim Lesen: {e}"})
                continue
            if not probe.strip():
                report.append({"file": name,
                               "status": "gespeichert, aber keine Glossarzeilen gefunden – "
                                         "Kopfzeile 'Frage | Antwort | Thema | Quelle' erwartet"})
                continue

            had = conn.execute("SELECT COUNT(*) FROM chunks WHERE source=?",
                               (name,)).fetchone()[0]
            if had:
                conn.execute("DELETE FROM chunks WHERE source=?", (name,))
                VECTORS[:] = [it for it in VECTORS if it["source"] != name]
                rebuild_bm25()

            before = len(report)
            _ingest_one(conn, name, str(dest), report, pending)
            if len(report) > before:
                report[-1]["status"] += (f" (Glossar aktualisiert, {had} alte Chunks ersetzt)"
                                         if had else " (Glossar)")
        conn.commit()
    finally:
        # get_db() haengt am Flask-Request-Kontext und wird per teardown_appcontext
        # geschlossen. Hier bewusst KEIN conn.close() — identisch zu ingest_folder().
        pass
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
        if not found and not (EVAL_DIR.exists() and any(EVAL_DIR.glob("*.xlsx"))):
            return jsonify({"report": [{"file": str(DATA_DIR),
                                        "status": "Keine PDF/DOCX/XLSX gefunden"}], "pending": []})
        for doc in found:
            if doc.name in existing:
                report.append({"file": doc.name, "status": "bereits indiziert – übersprungen"}); continue
            _ingest_one(conn, doc.name, str(doc), report, pending)

        # --- Glossare aus eval\ ---------------------------------------------
        # Bewusst IDEMPOTENT statt "überspringen": ein Glossar ist ein gepflegtes
        # Dokument, das sich aendert. Alte Chunks werden vorher entfernt, damit ein
        # erneutes Einlesen die Aenderungen uebernimmt, ohne Dubletten zu erzeugen.
        # (Dubletten wuerden df/avgdl im BM25 verfaelschen — also ALLE Scores, nicht
        # nur die des Glossars.)
        if EVAL_DIR.exists():
            for gx in sorted(EVAL_DIR.glob("*.xlsx"), key=lambda p: p.name.lower()):
                if _is_temp(gx.name):
                    continue
                had = conn.execute("SELECT COUNT(*) FROM chunks WHERE source=?",
                                   (gx.name,)).fetchone()[0]
                if had:
                    conn.execute("DELETE FROM chunks WHERE source=?", (gx.name,))
                    VECTORS[:] = [it for it in VECTORS if it["source"] != gx.name]
                    rebuild_bm25()
                before = len(report)
                _ingest_one(conn, gx.name, str(gx), report, pending)
                if len(report) > before and had:
                    report[-1]["status"] += f" (Glossar aktualisiert, {had} alte Chunks ersetzt)"
                elif len(report) > before:
                    report[-1]["status"] += " (Glossar)"
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
        conn.execute("UPDATE chunks SET vec=? WHERE id=?", (vec_to_blob(nv), cid))
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

# --------------------------------------------------------------- Eval-Harness (optional)
# Variante B: misst das ECHTE hybrid_search() gegen ein grosses Q/A-Set aus XLSX.
# Bewusst OPTIONAL eingebunden: fehlt eval_module.py, laeuft tiseR unveraendert
# weiter. Der Harness ist Messwerkzeug, kein Betriebsbestandteil — er darf den
# Produktivstart niemals verhindern.
# Registriert: /eval, /api/eval/load, /questions, /step, /report, /export, /compare
try:
    import eval_module
    eval_module.register(app, globals())
    log.info("Eval-Harness aktiv: http://localhost:%d/eval", PORT)
except ImportError:
    pass                      # eval_module.py nicht vorhanden -> Feature einfach aus
except Exception as _ee:
    log.warning("Eval-Harness nicht geladen (%s) — tiseR laeuft normal weiter.", _ee)

# --------------------------------------------------------------- Anonymisierung (optional)
# Markiert schuetzenswerte Stellen einer hochgeladenen Datei und ersetzt sie erst
# nach manueller Freigabe. Nutzt extract_text_any() und _xlsx_rows() aus diesem
# Modul; greift NICHT in Ingest, Index oder Retrieval ein — der Chat-Pfad bleibt
# unberuehrt. Wie der Eval-Harness bewusst OPTIONAL: fehlt anon_module.py, startet
# tiseR unveraendert.
# Registriert: /anon, /api/anon/config, /api/anon/scan
try:
    import anon_module
    anon_module.register(app, globals())
    log.info("Anonymisierung aktiv: http://localhost:%d/anon", PORT)
except ImportError:
    pass                      # anon_module.py nicht vorhanden -> Feature einfach aus
except Exception as _ae:
    log.warning("Anonymisierung nicht geladen (%s) — tiseR laeuft normal weiter.", _ae)

# ----------------------------------------------------------------
if __name__ == "__main__":
    import traceback, socket
    LOGFILE = APP_DIR / "tiser_error.log"

    def _fatal(msg):
        try:
            with open(LOGFILE, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n" + "-" * 60 + "\n")
        except Exception:
            pass
        print("\n" + msg)
        try:
            input("\n[Enter] zum Schliessen — Fehler steht auch in tiser_error.log")
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
        _fatal(f"Port {PORT} ist belegt — tiseR laeuft vermutlich schon.\n"
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

    log.info("tiseR laeuft: http://localhost:%d  (zum Beenden dieses Fenster schliessen)", PORT)
    try:
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except Exception:
        _fatal("FEHLER beim Serverstart:\n" + traceback.format_exc())
        raise SystemExit(1)
