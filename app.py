# -*- coding: utf-8 -*-
"""
armachat - RAG-Backend (AppLocker-konform, KEINE nativen DLLs)
Optimierungen ggue. v1:
  1. Strukturbewusstes Chunking (Ueberschriften-Schnitte, ToC/Kopfzeilen entfernt,
     Ueberschrift als Praefix im Chunk).
  2. Hybrid-Retrieval: Pure-Python BM25 (numpy-frei) + Cosine, fusioniert via
     Reciprocal Rank Fusion (RRF).
  3. PDF-Extraktion im LAYOUT-Modus (Tabellenspalten bleiben grob ausgerichtet).
  4. Zusaetzliche Formate DOCX + XLSX, geparst via stdlib zipfile + ElementTree
     (KEINE nativen DLLs -> AppLocker-konform; KEIN python-docx/openpyxl/lxml).
  5. /api/documents (Liste) + /api/delete_doc (Einzelloeschung) fuer die UI.
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
TOP_K        = 6        # etwas mehr Treffer, da Chunks jetzt kleiner sind
RRF_K        = 60       # RRF-Konstante (Standardwert)
PAGE_BREAK   = "\x0c"   # Seiten-/Foliengrenze -> harter Chunk-Schnitt

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
        text TEXT NOT NULL, vec TEXT)""")
    conn.commit()
    VECTORS.clear()
    for r in conn.execute("SELECT id, source, text, vec FROM chunks WHERE vec IS NOT NULL"):
        VECTORS.append({"id": r["id"], "source": r["source"], "text": r["text"],
                        "vec": json.loads(r["vec"]), "toks": _tok(r["text"])})
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
    return "\n".join(head + ["", body.strip()])

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
        head = []
        if sender:  head.append("Von: " + sender)
        if recips:  head.append("An: " + ", ".join(dict.fromkeys(recips)))
        if subject: head.append("Betreff: " + subject)
        return "\n".join(head + ["", body.strip()])
    finally:
        ole.close()

# Dispatcher nach Dateiendung. PdfReader/ZipFile/olefile akzeptieren Pfad ODER BytesIO.
_EXT_HANDLERS = {
    ".pdf": extract_pdf_text,   ".docx": extract_docx_text, ".xlsx": extract_xlsx_text,
    ".pptx": extract_pptx_text, ".csv": extract_csv_text,   ".txt": extract_txt_text,
    ".eml": extract_eml_text,   ".msg": extract_msg_text,
}
SUPPORTED_EXT = tuple(_EXT_HANDLERS.keys())

def extract_text_any(name: str, src) -> str:
    ext = Path(name).suffix.lower()
    handler = _EXT_HANDLERS.get(ext)
    if not handler:
        raise ValueError(f"Format '{ext or '?'}' nicht unterstuetzt "
                         f"(PDF, DOCX, XLSX, PPTX, CSV, TXT, EML, MSG)")
    return handler(src)

# --------------------------------------------------------------- Strukturbewusstes Chunking
_TOC  = re.compile(r'\.{5,}\s*\d+\s*$')
# Abschnittsnummern wie "2.1.3" — Komponenten max. 2-stellig, schliesst Daten
# (z. B. "12.05.2026") und 4-stellige Jahre aus.
_HEAD = re.compile(r'^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-ZÄÖÜ].{2,80})$')
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

def make_chunks(text: str, source: str):
    lines = _clean_lines(text)
    chunks = []; cur = []; head = ""; L = 0; ordn = 0
    def flush():
        nonlocal cur, L, ordn
        if cur:
            body = " ".join(cur).strip()
            if len(body) >= CHUNK_MIN:
                # Nur die Ueberschrift als Praefix (echte Struktur); KEIN Dateiname
                # mehr, der nur das Embedding-Fenster mit Rauschen fuellt.
                prefix = f"{head}: " if head else ""
                chunks.append({"source": source, "ord": ordn, "text": prefix + body})
                ordn += 1
        cur = []; L = 0
    for ln in lines:
        if ln == PAGE_BREAK:                         # Seiten-/Foliengrenze = harter Schnitt
            flush(); head = ""; continue             # Ueberschrift NICHT ueber Seiten schleppen
        m = _HEAD.match(ln)
        if m and re.search(r'[a-zäöü]', m.group(2)):  # echte Ueberschrift, kein Code-Fragment (R8 R2…)
            flush(); head = f"{m.group(1)} {m.group(2)}".strip()
            cur = [ln]; L = len(ln); continue
        if L + len(ln) > CHUNK_HARD: flush()
        cur.append(ln); L += len(ln) + 1
        if L >= CHUNK_TARGET and ln.endswith(('.', ';', ':')): flush()
    flush()
    return chunks

def _store_chunks(conn, chunks):
    pending = []; cur = conn.cursor()
    for c in chunks:
        cur.execute("INSERT INTO chunks(source, ord, text, vec) VALUES(?,?,?,NULL)",
                    (c["source"], c["ord"], c["text"]))
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

# --------------------------------------------------------------- Suche
def l2_normalize(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]

def _cosine_ranking(query_vec):
    q = l2_normalize(query_vec)
    scored = [(sum(a * b for a, b in zip(q, it["vec"])), idx) for idx, it in enumerate(VECTORS)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scored]

def _bm25_ranking(query_text):
    if not BM25 or not query_text: return []
    sc = BM25.scores(_tok(query_text))
    return sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)

def hybrid_search(query_vec, query_text, k=TOP_K):
    """Reciprocal Rank Fusion ueber Cosine- und BM25-Ranking."""
    if not VECTORS: return []
    rrf = {}
    if query_vec:
        for rank, idx in enumerate(_cosine_ranking(query_vec)):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(_bm25_ranking(query_text)):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    if not rrf:  # kein Vektor, kein Text -> nichts
        return []
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
    try: text = extract_text_any(name, src)
    except Exception as e: report.append({"file": name, "status": f"Lesefehler: {e}"}); return
    if not text or not text.strip():
        report.append({"file": name, "status": "LEER (kein Textlayer?)"}); return
    chunks = make_chunks(text, name)
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
        row = conn.execute("SELECT source, text FROM chunks WHERE id=?", (cid,)).fetchone()
        if row:
            VECTORS.append({"id": cid, "source": row["source"], "text": row["text"],
                            "vec": nv, "toks": _tok(row["text"])})
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
