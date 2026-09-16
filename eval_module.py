# -*- coding: utf-8 -*-
"""
tiseR - Eval-Modul (Variante B)   [Version 20260904v01]

ZWECK
  Misst das ECHTE Hybrid-Retrieval (Cosine + BM25 -> RRF) gegen ein grosses
  Frage/Antwort-Set aus einer XLSX-Datei. Ersetzt das 22-Fragen-Golden-Set,
  das statistisch nicht belastbar ist.

WARUM ZWEITEILIG
  Query-Embeddings entstehen im Browser (Transformers.js). In Python gibt es
  auf dem Bundesrechner kein onnxruntime (AppLocker). Der Ablauf ist deshalb:

    static/eval.html  --(1) holt Frage-------------------->  /api/eval/questions
                      --(2) embed_worker.js -> Vektor
                      --(3) sendet {qid, vector}---------->  /api/eval/step
                                                              | hybrid_search()
                                                              | scoring
                      <-(4) Einzelergebnis-----------------|
                      --(5) am Ende------------------------>  /api/eval/report

  Schritt 3 ruft DIESELBE hybrid_search() auf, die auch /api/search nutzt.
  Damit misst der Harness den Produktionspfad, keine Nachbildung.

SCORING
  Das XLSX enthaelt formulierte Antworten, keine kurzen Gold-Tokens. Ein
  simples "steht das Wort drin?" ist damit nicht moeglich. Stattdessen:
    - Aus der Antwort werden INHALTSTRAGENDE Terme gezogen (>=4 Zeichen,
      keine Stoppwoerter) plus alle ZAHLEN/Betraege (die sind bei Fristen und
      Schwellenwerten das eigentliche Ziel).
    - Jeder Term wird mit dem BM25-IDF DES KORPUS gewichtet. Seltene Fachbegriffe
      zaehlen also mehr als "Bundesverwaltung". Ohne diese Gewichtung wuerde
      jeder Chunk, der zufaellig viel Amtsdeutsch enthaelt, gut aussehen.
    - coverage = gewichteter Anteil der Terme, der in den Top-k-Chunks vorkommt.
    - Treffer, wenn coverage >= HIT_COVERAGE.

  DAS IST EINE HEURISTIK, KEIN GROUND TRUTH. Sie misst zuverlaessig
  VERAENDERUNGEN (vorher/nachher), nicht absolute Qualitaet. Genau dafuer wird
  sie gebraucht. Absolutwerte nicht ueberinterpretieren.

EINBINDUNG in app.py  (zwei Zeilen, ans Ende der Routen-Sektion):
    import eval_module
    eval_module.register(app, globals())
"""
import io, json, math, re, zipfile, time
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import request, jsonify, send_from_directory

# --------------------------------------------------------------- Parameter
HIT_COVERAGE   = 0.45   # ab diesem gewichteten Term-Anteil gilt eine Frage als getroffen
MIN_TERM_LEN   = 4      # kuerzere Woerter tragen kaum Bedeutung
MAX_TERMS      = 25     # lange Antworten nicht ueberproportional gewichten
DEFAULT_TOPK   = 6      # identisch zu TOP_K in app.py; ueberschreibbar per Request

# Bewusst knappe Stoppwortliste. Zu aggressiv filtern wuerde Fachbegriffe
# entfernen ("Stelle", "Frist" sind hier INHALT, nicht Fuellwort).
_STOP = set("""
aber alle allen aller alles als also andere anderen auch auf aus bei beim
dass dazu dann der die das dem den des dieser diese dieses dort durch eine
einen einem eines einer eine fuer für hat haben hier ihre ihrer ist kann
kein keine koennen können mehr muss müssen nach nicht noch nur oder sein
sich sind sowie über und unter vom von vor wenn werden wird wurde zum zur
zwischen dabei damit daher jedoch sowohl bzw etwa insbesondere gemaess
gemäss grundsaetzlich grundsätzlich regel falls sofern soweit welche welcher
""".split())

_RX_TOK  = re.compile(r"[\wäöüÄÖÜàéèç./'-]+", re.UNICODE)
_RX_NUM  = re.compile(r"\d")

_STATE = {"rows": [], "path": None, "results": {}, "started": None}


# --------------------------------------------------------------- XLSX (stdlib)
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col_index(ref: str) -> int:
    """'C7' -> 2 (0-basiert). Noetig, weil leere Zellen in der XML fehlen
    koennen und Spalten sonst verrutschen."""
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return n - 1


def _cell_text(c, shared):
    typ = c.get("t")
    if typ == "s":
        v = c.find(_S + "v")
        if v is not None and v.text is not None:
            try:
                return shared[int(v.text)]
            except (ValueError, IndexError):
                return ""
    if typ == "inlineStr":
        isn = c.find(_S + "is")
        if isn is not None:
            return "".join(t.text or "" for t in isn.iter(_S + "t"))
    if typ == "str":
        v = c.find(_S + "f/..") if False else c.find(_S + "v")
        return (v.text or "") if v is not None else ""
    v = c.find(_S + "v")
    return (v.text or "") if v is not None else ""


def read_qa_xlsx(path):
    """Liest ein Arbeitsblatt-Set mit Kopfzeile Frage|Antwort|Thema|Quelle.
    Liefert [{qid, frage, antwort, thema, quelle, sheet}].
    Reine stdlib (zipfile + ElementTree) -> AppLocker-konform, wie app.py."""
    rows = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()

        shared = []
        if "xl/sharedStrings.xml" in names:
            with z.open("xl/sharedStrings.xml") as f:
                sroot = ET.parse(f).getroot()
            for si in sroot.iter(_S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(_S + "t")))

        # Blattnamen aus workbook.xml + rels, damit die Reihenfolge stimmt und
        # echte Namen (BPG, Kommerzglossar, ...) statt sheet1.xml erscheinen.
        sheet_map = []
        try:
            with z.open("xl/workbook.xml") as f:
                wbroot = ET.parse(f).getroot()
            rels = {}
            with z.open("xl/_rels/workbook.xml.rels") as f:
                rroot = ET.parse(f).getroot()
            for rel in rroot:
                rels[rel.get("Id")] = rel.get("Target")
            for sh in wbroot.iter(_S + "sheet"):
                tgt = rels.get(sh.get(_R + "id"), "")
                tgt = tgt.lstrip("/")
                if not tgt.startswith("xl/"):
                    tgt = "xl/" + tgt
                sheet_map.append((sh.get("name"), tgt))
        except Exception:
            sheet_map = [(n.split("/")[-1], n) for n in sorted(names)
                         if n.startswith("xl/worksheets/") and n.endswith(".xml")]

        qid = 0
        for sheet_name, target in sheet_map:
            if target not in names:
                continue
            with z.open(target) as f:
                wroot = ET.parse(f).getroot()
            header_done = False
            for row in wroot.iter(_S + "row"):
                vals = {}
                for c in row.iter(_S + "c"):
                    ref = c.get("r") or ""
                    vals[_col_index(ref)] = (_cell_text(c, shared) or "").strip()
                if not vals:
                    continue
                get = lambda i: vals.get(i, "")
                if not header_done:
                    header_done = True
                    if get(0).lower().startswith("frage"):
                        continue          # Kopfzeile ueberspringen
                frage, antwort = get(0), get(1)
                if not frage or not antwort:
                    continue
                rows.append({"qid": qid, "frage": frage, "antwort": antwort,
                             "thema": get(2), "quelle": get(3), "sheet": sheet_name})
                qid += 1
    return rows


# --------------------------------------------------------------- Scoring
def gold_terms(answer: str):
    """Inhaltstragende Terme einer Musterantwort.
    Zahlen werden IMMER behalten: bei Fristen ('drei Monate', '72 Stunden') und
    Schwellenwerten ist die Zahl das eigentliche Pruefkriterium."""
    seen, out = set(), []
    for t in _RX_TOK.findall(answer.lower()):
        t = t.strip("./'-")
        if not t or t in seen:
            continue
        has_num = bool(_RX_NUM.search(t))
        if not has_num:
            if len(t) < MIN_TERM_LEN or t in _STOP:
                continue
        seen.add(t)
        out.append(t)
    return out[:MAX_TERMS]


def score_question(answer: str, chunk_texts, bm25=None, tok=None):
    """Gewichtete Term-Abdeckung der Musterantwort in den Top-k-Chunks.
    Rueckgabe (coverage, hit, first_rank, n_terms).
    first_rank = 1-basierter Rang des ersten Chunks, der etwas beitraegt (fuer MRR)."""
    terms = gold_terms(answer)
    if not terms:
        return 0.0, False, 0, 0

    # IDF-Gewichtung aus dem ECHTEN Korpus. Fehlt BM25 (leerer Index), fallen
    # alle Gewichte auf 1.0 -> reine ungewichtete Abdeckung.
    def w(t):
        if bm25 is None:
            return 1.0
        # Unbekannte Terme sind maximal selten -> hohes Gewicht, gedeckelt.
        return min(bm25.idf.get(t, 6.0), 8.0) or 1.0

    weights = {t: w(t) for t in terms}
    total = sum(weights.values()) or 1.0

    covered, first_rank = set(), 0
    for rank, txt in enumerate(chunk_texts, start=1):
        low = (txt or "").lower()
        toks = set(_RX_TOK.findall(low))
        newly = [t for t in terms if t not in covered and (t in toks or t in low)]
        if newly and not first_rank:
            first_rank = rank
        covered.update(newly)

    cov = sum(weights[t] for t in covered) / total
    return cov, cov >= HIT_COVERAGE, first_rank, len(terms)


# --------------------------------------------------------------- Aggregation
def aggregate(results, rows):
    by_sheet = {}
    for r in results.values():
        row = rows[r["qid"]]
        b = by_sheet.setdefault(row["sheet"], {"n": 0, "hits": 0, "cov": 0.0, "rr": 0.0})
        b["n"] += 1
        b["hits"] += 1 if r["hit"] else 0
        b["cov"] += r["coverage"]
        b["rr"] += (1.0 / r["first_rank"]) if r["first_rank"] else 0.0

    sheets = []
    for name, b in sorted(by_sheet.items()):
        sheets.append({
            "sheet": name, "n": b["n"], "hits": b["hits"],
            "recall": round(b["hits"] / b["n"], 4) if b["n"] else 0.0,
            "coverage": round(b["cov"] / b["n"], 4) if b["n"] else 0.0,
            "mrr": round(b["rr"] / b["n"], 4) if b["n"] else 0.0,
        })
    tot_n = sum(s["n"] for s in sheets)
    tot_h = sum(s["hits"] for s in sheets)
    overall = {
        "n": tot_n, "hits": tot_h,
        "recall": round(tot_h / tot_n, 4) if tot_n else 0.0,
        "coverage": round(sum(s["coverage"] * s["n"] for s in sheets) / tot_n, 4) if tot_n else 0.0,
        "mrr": round(sum(s["mrr"] * s["n"] for s in sheets) / tot_n, 4) if tot_n else 0.0,
    }
    return {"overall": overall, "sheets": sheets}


# --------------------------------------------------------------- Flask-Routen
def register(app, ctx):
    """ctx = globals() von app.py. Wir greifen bewusst LAZY darauf zu, damit
    VECTORS/BM25 zum Aufrufzeitpunkt gelesen werden, nicht beim Import."""
    APP_DIR = ctx["APP_DIR"]
    EVAL_DIR = APP_DIR / "eval"

    def _hybrid():
        return ctx["hybrid_search"]

    def _bm25():
        return ctx.get("BM25")

    @app.route("/eval")
    def eval_page():
        return send_from_directory(ctx["STATIC_DIR"], "eval.html")

    @app.route("/api/eval/load", methods=["POST"])
    def eval_load():
        """Laedt das XLSX aus eval\\ (oder per Upload) und setzt den Lauf zurueck."""
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        up = request.files.get("file")
        if up is not None and up.filename:
            path = EVAL_DIR / Path(up.filename).name
            up.save(str(path))
        else:
            data = request.get_json(force=True, silent=True) or {}
            wanted = data.get("path")
            if wanted:
                path = Path(wanted)
                if not path.is_absolute():
                    path = EVAL_DIR / wanted
            else:
                cands = sorted(EVAL_DIR.glob("*.xlsx"))
                if not cands:
                    return jsonify({"error": f"Keine .xlsx in {EVAL_DIR} gefunden."}), 400
                path = cands[0]
        try:
            rows = read_qa_xlsx(path)
        except Exception as e:
            return jsonify({"error": f"XLSX nicht lesbar: {e}"}), 400
        if not rows:
            return jsonify({"error": "Keine Frage/Antwort-Zeilen gefunden "
                                     "(Kopfzeile 'Frage | Antwort | Thema | Quelle' erwartet)."}), 400
        _STATE["rows"] = rows
        _STATE["path"] = str(path)
        _STATE["results"] = {}
        _STATE["started"] = time.time()
        counts = {}
        for r in rows:
            counts[r["sheet"]] = counts.get(r["sheet"], 0) + 1
        return jsonify({"ok": True, "path": str(path), "total": len(rows),
                        "sheets": [{"sheet": k, "n": v} for k, v in counts.items()],
                        "indexed_chunks": len(ctx["VECTORS"])})

    @app.route("/api/eval/questions")
    def eval_questions():
        """Fragen des Laufs. sheet= filtert, limit= kuerzt (Probelauf)."""
        sheet = (request.args.get("sheet") or "").strip()
        try:
            limit = int(request.args.get("limit") or 0)
        except ValueError:
            limit = 0
        rows = _STATE["rows"]
        if not rows:
            return jsonify({"error": "Kein Eval-Set geladen."}), 400
        sel = [r for r in rows if not sheet or r["sheet"] == sheet]
        if limit > 0:
            sel = sel[:limit]
        return jsonify({"questions": [{"qid": r["qid"], "frage": r["frage"],
                                       "sheet": r["sheet"]} for r in sel]})

    @app.route("/api/eval/step", methods=["POST"])
    def eval_step():
        """Kern: fuehrt das ECHTE hybrid_search() aus und bewertet eine Frage."""
        data = request.get_json(force=True, silent=True) or {}
        qid, vec = data.get("qid"), data.get("vector")
        if qid is None or qid >= len(_STATE["rows"]):
            return jsonify({"error": "Unbekannte qid."}), 400
        try:
            k = int(data.get("k") or DEFAULT_TOPK)
        except ValueError:
            k = DEFAULT_TOPK
        row = _STATE["rows"][qid]

        top = _hybrid()(vec, row["frage"], k)
        texts = [it["text"] for _s, it in top]
        srcs = [it["source"] for _s, it in top]
        cov, hit, rank, nterms = score_question(row["antwort"], texts, _bm25())

        _STATE["results"][qid] = {"qid": qid, "coverage": round(cov, 4), "hit": hit,
                                  "first_rank": rank, "terms": nterms,
                                  "sources": srcs[:3]}
        return jsonify({"qid": qid, "coverage": round(cov, 4), "hit": hit,
                        "first_rank": rank, "sheet": row["sheet"],
                        "top_source": srcs[0] if srcs else None})

    @app.route("/api/eval/report")
    def eval_report():
        if not _STATE["results"]:
            return jsonify({"error": "Noch keine Ergebnisse."}), 400
        rep = aggregate(_STATE["results"], _STATE["rows"])
        rep["path"] = _STATE["path"]
        rep["indexed_chunks"] = len(ctx["VECTORS"])
        rep["hit_coverage"] = HIT_COVERAGE
        rep["duration_s"] = round(time.time() - (_STATE["started"] or time.time()), 1)
        return jsonify(rep)

    @app.route("/api/eval/export", methods=["POST"])
    def eval_export():
        """Schreibt Report + Einzelergebnisse als JSON nach eval\\.
        Baseline-Datei fuer den Vorher/Nachher-Vergleich."""
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        data = request.get_json(force=True, silent=True) or {}
        label = re.sub(r"[^\w.-]", "_", (data.get("label") or "run"))[:60]
        rep = aggregate(_STATE["results"], _STATE["rows"])
        rep["label"] = label
        rep["path"] = _STATE["path"]
        rep["indexed_chunks"] = len(ctx["VECTORS"])
        rep["hit_coverage"] = HIT_COVERAGE
        rep["details"] = [
            dict(_STATE["results"][q], sheet=_STATE["rows"][q]["sheet"],
                 frage=_STATE["rows"][q]["frage"])
            for q in sorted(_STATE["results"])
        ]
        out = EVAL_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{label}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True, "file": str(out), "overall": rep["overall"]})

    @app.route("/api/eval/compare")
    def eval_compare():
        """Vergleicht die zwei juengsten Exporte pro Blatt. Das ist der
        eigentliche Zweck: die DIFFERENZ, nicht der Absolutwert."""
        files = sorted(EVAL_DIR.glob("eval_*.json"))
        if len(files) < 2:
            return jsonify({"error": "Mindestens zwei Exporte noetig."}), 400
        a = json.loads(files[-2].read_text(encoding="utf-8"))
        b = json.loads(files[-1].read_text(encoding="utf-8"))
        amap = {s["sheet"]: s for s in a["sheets"]}
        diff = []
        for s in b["sheets"]:
            prev = amap.get(s["sheet"])
            if not prev:
                continue
            diff.append({"sheet": s["sheet"], "n": s["n"],
                         "recall_before": prev["recall"], "recall_after": s["recall"],
                         "delta": round(s["recall"] - prev["recall"], 4),
                         "mrr_before": prev["mrr"], "mrr_after": s["mrr"]})
        diff.sort(key=lambda d: d["delta"])
        return jsonify({"before": files[-2].name, "after": files[-1].name,
                        "overall_before": a["overall"], "overall_after": b["overall"],
                        "sheets": diff})

    return app
