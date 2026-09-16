# -*- coding: utf-8 -*-
"""
tiseR - Anonymisierungs-Modul   [Version 20260908v02]

ZWECK
  Eine hochgeladene Datei auf schuetzenswerte Inhalte pruefen, die Funde
  MARKIEREN und erst nach manueller Freigabe ersetzen. Ziel ist ein Text, der
  in einem oeffentlichen Chat-Dienst weiterverarbeitet werden kann.

  WICHTIG - was dieses Modul NICHT leistet:
  Es entscheidet NICHT ueber die Klassifizierung. Ein anonymisiertes INTERN-
  oder VERTRAULICH-Dokument bleibt eingestuft. Die Entfernung von Namen macht
  ein Dokument nicht freigabefaehig. Diese Pruefung bleibt beim Menschen.

  Ebenso NICHT geleistet: Schutz vor Re-Identifikation. "[PERSON_01], Leiter
  Projekt X bei armasuisse" identifiziert eindeutig, auch ohne Namen.

ZWEI QUELLEN, BEWUSST GETRENNT
  1. anon_patterns.json  - STRUKTURMUSTER (AHV, IBAN, Koordinaten, Dok-IDs).
     Versionierbar, unbedenklich, gehoert ins Repo.
  2. Blatt "_anon" in der Glossar-XLSX unter eval\\ - KONKRETE BEGRIFFE
     (Namen, Projektbezeichnungen, Standorte). Der Unterstrich-Praefix nutzt
     GLOSSARY_SHEET_SKIP: das Blatt wird von extract_xlsx_glossary_text()
     uebersprungen und gelangt NIE in den Index. Die Datei liegt lokal und
     darf niemals publiziert werden - eine sauber gepflegte Liste aller
     schuetzenswerten Begriffe ist ein hoeherwertiges Ziel als jedes einzelne
     Dokument, aus dem sie stammt.

     Spalten:  Begriff | Typ | Ersatz
       Typ = wort   -> Wortgrenzen, case-insensitive (Standard)
             teil   -> Teilstring (Projektkuerzel, Systemnamen)
             regex  -> eigener regulaerer Ausdruck

PRUEFZIFFERN
  AHV (EAN-13) und IBAN (mod 97) werden validiert, nicht nur gematcht. Ohne
  das erzeugt jede 13-stellige Zahlenfolge einen Falschtreffer und die
  Trefferliste wird unbrauchbar.

LLM-ANTEIL
  Laeuft NICHT hier. Das Modell liegt im Browser (llm_worker.js), Python kann
  es nicht aufrufen. Dieses Modul liefert nur die gefensterten Textabschnitte;
  anon.html schickt sie durch das Modell und traegt die Funde zurueck.

  Das Modell gibt ausschliesslich eine LISTE aus, es schreibt den Text nicht
  um: bei PROMPT_BUDGET=1800 passt ein Dokument nicht in Ein- plus Ausgabe,
  und generatives Umschreiben verfaelscht Zahlen.

EINBINDUNG in app.py (zwei Zeilen, analog eval_module):
    import anon_module
    anon_module.register(app, globals())
"""
import io, json, re
from pathlib import Path
from flask import request, jsonify, send_from_directory

VERSION = "20260908v02"

WINDOW_CHARS  = 1400   # Fenstergroesse fuer den LLM-Durchlauf
WINDOW_OVER   = 200    # Ueberlappung, damit Namen an Fenstergrenzen nicht verloren gehen
ANON_SHEET    = "_anon"


# --------------------------------------------------------------- Pruefziffern
def _chk_ean13(s: str) -> bool:
    """AHV-Nummer: 13 Ziffern, EAN-13-Pruefziffer."""
    d = re.sub(r"\D", "", s)
    if len(d) != 13:
        return False
    tot = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(d[:12]))
    return (10 - tot % 10) % 10 == int(d[12])


def _chk_iban(s: str) -> bool:
    """IBAN: mod-97 == 1."""
    t = re.sub(r"[^A-Za-z0-9]", "", s).upper()
    if len(t) < 15 or len(t) > 34:
        return False
    t = t[4:] + t[:4]
    num = "".join(str(ord(c) - 55) if c.isalpha() else c for c in t)
    try:
        return int(num) % 97 == 1
    except ValueError:
        return False


def _chk_lv95(s: str) -> bool:
    """LV95 liegt im Rechteck E 2'480'000-2'840'000 / N 1'070'000-1'300'000.
    Ohne diese Bereichspruefung matcht jede Zahlenpaarung mit fuehrender 2/1."""
    n = re.findall(r"[\d]+(?:\.\d+)?", s.replace("'", "").replace(".", "", 0))
    try:
        parts = re.split(r"[\s,/]+", s.strip())
        e = float(re.sub(r"[^\d.]", "", parts[0].replace("'", "")))
        nn = float(re.sub(r"[^\d.]", "", parts[-1].replace("'", "")))
    except (ValueError, IndexError):
        return False
    return 2480000 <= e <= 2840000 and 1070000 <= nn <= 1300000


def _chk_lv03(s: str) -> bool:
    """LV03: E 480'000-840'000 / N 70'000-300'000."""
    try:
        parts = re.split(r"[\s,/]+", s.strip())
        e = float(re.sub(r"[^\d.]", "", parts[0].replace("'", "")))
        nn = float(re.sub(r"[^\d.]", "", parts[-1].replace("'", "")))
    except (ValueError, IndexError):
        return False
    return 480000 <= e <= 840000 and 70000 <= nn <= 300000


_CHECKS = {"ean13": _chk_ean13, "iban": _chk_iban, "lv95": _chk_lv95, "lv03": _chk_lv03}


# --------------------------------------------------------------- Musterdatei
def _load_patterns(app_dir: Path):
    """Laedt anon_patterns.json und prueft jedes Muster gegen sein 'beispiel'.
    Ein Tippfehler im Regex faellt so sofort auf, statt still nichts zu finden."""
    path = app_dir / "anon_patterns.json"
    if not path.exists():
        return [], [f"anon_patterns.json fehlt ({path})"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [], [f"anon_patterns.json unlesbar: {e}"]

    pats, warn = [], []
    for m in data.get("muster", []):
        if not m.get("aktiv", True):
            continue
        name = m.get("name") or "?"
        try:
            rx = re.compile(m["regex"])
        except Exception as e:
            warn.append(f"Muster '{name}': ungueltiger Regex ({e}) - uebersprungen")
            continue
        bsp = m.get("beispiel")
        if bsp and not rx.search(bsp):
            warn.append(f"Muster '{name}': Beispiel '{bsp}' wird vom eigenen Regex NICHT erkannt")
        chk = m.get("pruefung")
        if chk and chk not in _CHECKS:
            warn.append(f"Muster '{name}': unbekannte Pruefung '{chk}' - ignoriert")
            chk = None
        pats.append({"name": name, "rx": rx, "chk": chk,
                     "ersatz": m.get("ersatz") or "[GESCHWAERZT]"})
    return pats, warn


# --------------------------------------------------------------- Begriffsliste
def _load_terms(ctx):
    """Liest Blatt '_anon' aus jeder XLSX unter eval\\.
    Spalten: Begriff | Typ | Ersatz. Fehlt das Blatt, ist die Liste leer -
    das Modul funktioniert dann nur mit Strukturmustern."""
    app_dir = ctx["APP_DIR"]
    eval_dir = app_dir / "eval"
    rows_fn = ctx.get("_xlsx_rows")
    terms, warn = [], []
    if rows_fn is None or not eval_dir.exists():
        return terms, warn

    for xf in sorted(eval_dir.glob("*.xlsx")):
        if xf.name.startswith("~$"):
            continue
        try:
            first = True
            for sheet, vals in rows_fn(xf):
                if not str(sheet or "").lower().startswith(ANON_SHEET):
                    continue
                get = lambda i: (vals[i] if i < len(vals) else "").strip()
                begriff, typ, ersatz = get(0), get(1).lower(), get(2)
                if first and begriff.lower().startswith("begriff"):
                    first = False
                    continue
                first = False
                if not begriff:
                    continue
                typ = typ if typ in ("wort", "teil", "regex") else "wort"
                if typ == "wort":
                    rx = re.compile(r"\b" + re.escape(begriff) + r"\b", re.I)
                elif typ == "teil":
                    rx = re.compile(re.escape(begriff), re.I)
                else:
                    try:
                        rx = re.compile(begriff, re.I)
                    except Exception as e:
                        warn.append(f"Begriff-Regex '{begriff[:30]}' ungueltig ({e})")
                        continue
                terms.append({"name": f"Begriffsliste ({typ})", "rx": rx, "chk": None,
                              "ersatz": ersatz or "[BEGRIFF]"})
        except Exception as e:
            warn.append(f"{xf.name}: Blatt '{ANON_SHEET}' nicht lesbar ({e})")
    return terms, warn


# --------------------------------------------------------------- Trefferlogik
def _find(text: str, rules):
    """Alle Regeln anwenden. Ueberlappende Treffer: der laengere gewinnt -
    sonst schwaerzt ein kurzes Muster mitten in einer laengeren IBAN."""
    raw = []
    for r in rules:
        for m in r["rx"].finditer(text):
            frag = m.group(0)
            if r["chk"] and not _CHECKS[r["chk"]](frag):
                continue
            raw.append({"start": m.start(), "end": m.end(), "text": frag,
                        "muster": r["name"], "ersatz": r["ersatz"], "quelle": "muster"})
    raw.sort(key=lambda h: (h["start"], -(h["end"] - h["start"])))
    out, last = [], -1
    for h in raw:
        if h["start"] < last:
            continue
        out.append(h)
        last = h["end"]
    return out


def _windows(text: str):
    """Text in ueberlappende Fenster schneiden, moeglichst an Satzgrenzen."""
    wins, i, n = [], 0, len(text)
    while i < n:
        j = min(i + WINDOW_CHARS, n)
        if j < n:
            cut = max(text.rfind(". ", i + WINDOW_CHARS // 2, j),
                      text.rfind("\n", i + WINDOW_CHARS // 2, j))
            if cut > i:
                j = cut + 1
        wins.append({"index": len(wins), "start": i, "text": text[i:j]})
        if j >= n:
            break
        i = max(j - WINDOW_OVER, i + 1)
    return wins


# --------------------------------------------------------------- Registrierung
def register(app, ctx):
    """ctx = globals() von app.py. Zugriff bewusst lazy."""
    APP_DIR = ctx["APP_DIR"]

    @app.route("/anon")
    def anon_page():
        return send_from_directory(ctx["STATIC_DIR"], "anon.html")

    @app.route("/api/anon/config")
    def anon_config():
        """Zeigt, was geladen wurde. Erster Blick bei 'findet nichts'."""
        pats, w1 = _load_patterns(APP_DIR)
        terms, w2 = _load_terms(ctx)
        return jsonify({
            "version": VERSION,
            "muster": [p["name"] for p in pats],
            "begriffe": len(terms),
            "warnungen": w1 + w2,
        })

    @app.route("/api/anon/scan", methods=["POST"])
    def anon_scan():
        """Datei -> Text -> Musterfunde + LLM-Fenster.
        Es wird NICHTS ersetzt und NICHTS gespeichert; die Ersetzung passiert
        im Browser nach manueller Freigabe."""
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "Keine Datei uebermittelt"}), 400
        try:
            text, _seed = ctx["extract_text_any"](f.filename, io.BytesIO(f.read()))
        except Exception as e:
            return jsonify({"error": f"{f.filename}: {e}"}), 400
        if not (text or "").strip():
            return jsonify({"error": f"{f.filename}: kein extrahierbarer Text"}), 400

        # ZEILENENDEN NORMALISIEREN - nicht kosmetisch, sondern zwingend.
        # MSG/EML liefern \r\n. Sobald der Text im Browser per innerHTML ins DOM
        # geht, normalisiert der HTML-Parser \r\n zu \n: jedes \r verschwindet aus
        # den Textknoten. Eine im DOM gemessene Auswahlposition waere dann pro
        # vorangehender Zeile um ein Zeichen zu klein, und die manuelle Markierung
        # traefe eine voellig andere Stelle. Wird hier normalisiert, stimmen
        # Backend-Offsets, DOC.text und DOM-Offsets ueberein.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        pats, w1 = _load_patterns(APP_DIR)
        terms, w2 = _load_terms(ctx)
        hits = _find(text, pats + terms)
        return jsonify({
            "version": VERSION,
            "filename": f.filename,
            "text": text,
            "hits": hits,
            "windows": _windows(text),
            "warnungen": w1 + w2,
            "stats": {"zeichen": len(text), "treffer": len(hits),
                      "muster_aktiv": len(pats), "begriffe_aktiv": len(terms)},
        })
