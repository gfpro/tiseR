# tiseR — tiny secure RAG

Lokales RAG-/Dokumentenintelligenz-System für Arbeitsplätze ohne Adminrechte,
ohne Docker und ohne Internetzugang zur Laufzeit. Dokumente → strukturbewusste
Chunks → Embedding im Browser → Hybrid-Retrieval in Python → Antwort eines
lokalen LLM ausschliesslich aus dem gefundenen Kontext.

**Alles läuft auf dem eigenen Rechner. Es verlässt kein Byte die Maschine.**

---

## Warum diese Architektur so aussieht, wie sie aussieht

Die Zielumgebung (Bundesrechner unter AppLocker-Whitelist) verbietet nativen
Code. Das ist keine Vorsichtsmassnahme, sondern eine harte Grenze:

| Blockiert | Grund |
|---|---|
| `onnxruntime` (Python), `torch`, `faiss` | native `.pyd`/`.dll` |
| `pdfminer.six` | Abhängigkeit `cryptography` (nativ) |
| `rank_bm25` | Abhängigkeit `numpy` (nativ) |
| `.bat`-Dateien aus nicht-freigegebenen Pfaden | AppLocker-Regel |
| `webbrowser.open()`, COM/PowerShell-.NET-Aufrufe | prozessgestarteter Fremdcode |
| Docker | nicht installierbar |

**Konsequenz:** Die gesamte ML-Inferenz läuft im Browser über Transformers.js
(ONNX Runtime Web, WebGPU mit WASM-Fallback). Python macht nur Dateiparsing,
Chunking, Suche und HTTP — mit reiner Standardbibliothek plus drei Wheels.

Alles, was im Code seltsam aussieht, hat hier seine Ursache: DOCX/XLSX/PPTX
werden mit `zipfile` + `xml.etree` gelesen statt mit `python-docx`/`openpyxl`,
BM25 ist von Hand in reinem Python implementiert, und der Start erfolgt über
eine manuell erstellte `.lnk`.

---

## Installation

### 1. Python

Python 3.10 aus dem Software-Kiosk. Prüfen:

```
python --version
```

### 2. Repo ablegen

Nach `C:\tiseR\`. **Achtung beim Laufwerkswechsel in CMD:** `cd C:\tiseR` von
`H:\` aus wechselt das Laufwerk *nicht*. Korrekt ist:

```
cd /d C:\tiseR
```

### 3. Abhängigkeiten (offline)

Auf einem Rechner **mit** Internet:

```
pip download -r requirements.txt -d wheels
```

Auf dem Zielrechner:

```
cd /d C:\tiseR
pip install --no-index --find-links wheels -r requirements.txt
```

Nur gebaute `.whl`-Dateien funktionieren — Source-Distributionen (`.tar.gz`)
scheitern, weil sie einen Build-Schritt bräuchten. Fehlende transitive
Abhängigkeiten tauchen einzeln auf; sie gehören danach in `requirements.txt`.

Der Ordner `wheels\` ist im Repo eingecheckt, damit Schritt 1 entfallen kann.

### 4. Transformers.js

Liegt als `static\transformersjs-420\` im Repo. Diese Dateien stammen aus den
npm-Paketen `@huggingface/transformers` und `onnxruntime-web` — beide werden
als `.tgz` über npm verteilt, nicht über PyPI. Die exakte
`onnxruntime-web`-Version steht in `package/package.json` unter `dependencies`.

**Sie sind absichtlich eingecheckt**, weil Dev-Releases von `onnxruntime-web`
aus der npm-Registry verschwinden können.

### 5. Modelle

Modelle liegen **nicht** im Repo (zu gross). Sie müssen manuell nach
`C:\tiseR\models\` kopiert werden. Details unten.

### 6. Start

```
cd /d C:\tiseR
python app.py
```

Dann in Edge: **http://localhost:8000**

> **Der häufigste Fehler:** `index.html` doppelklicken. Unter `file://` darf der
> Browser keine Web-Worker starten („SecurityError … origin 'null'"). Immer über
> den Server.

Für einen Doppelklick-Start: Rechtsklick auf Desktop → Neu → Verknüpfung,
Ziel `C:\...\python.exe`, Argument `"C:\tiseR\app.py"`, Arbeitsverzeichnis
`C:\tiseR`. Das ist unter AppLocker der einzige funktionierende Weg —
`.bat`-Dateien und programmatisch erzeugte Verknüpfungen scheitern.

---

## Ordnerstruktur

```
C:\tiseR\
├─ app.py                  Flask-Backend, Chunking, Hybrid-Suche
├─ index.html              UI (liegt im HAUPTordner, nicht in static\)
├─ eval_harness.py         Retrieval-Messung gegen das Golden-Set
├─ requirements.txt
├─ README.md
├─ armachat.db             entsteht beim 1. Start  (Umbenennung offen, s. u.)
├─ data\                   PDFs etc. für "Ordner data\ einlesen"
├─ wheels\                 Offline-Installation
├─ static\
│  ├─ embed_worker.js
│  ├─ llm_worker.js
│  └─ transformersjs-420\
└─ models\
   ├─ multilingual-e5-large\      Embedder — PFLICHT
   ├─ LFM2-2.6B-Bund\             LLM (Fine-Tune, primär)
   └─ <weitere LLMs>\             erscheinen automatisch im Dropdown
```

---

## Modelle beschaffen

### Embedder (Pflicht)

`Xenova/multilingual-e5-large` — 1024 Dimensionen, int8.

Nach `models\multilingual-e5-large\` gehören genau diese Dateien:

```
multilingual-e5-large\
├─ onnx\
│  └─ model_quantized.onnx      ~562 MB   (selbstenthalten, kein External Data)
├─ config.json
├─ tokenizer.json
├─ tokenizer_config.json
├─ special_tokens_map.json
└─ sentencepiece.bpe.model
```

**Nicht** herunterladen: `model.onnx`, `model.onnx_data`, `model_fp16.onnx` —
zusammen über 3,9 GB, die nie angefragt werden.

Die Tokenizer-Dateien liegen im **Hauptordner**, nur die `.onnx` in `onnx\`.
Der Ordnername muss exakt der Konstante `EMBED_MODEL` in
`static\embed_worker.js` entsprechen, sonst gibt es einen 404 auf `config.json`
und der Embedder lädt nie.

> **Instruct-Variante:** Es existiert auch `multilingual-e5-large-instruct`.
> Die beiden sind architektonisch identisch, brauchen aber unterschiedliche
> Query-Präfixe (`query: ` vs. eine Instruct-Vorlage). Beim Modellwechsel muss
> `QUERY_MODE` in `embed_worker.js` mitgezogen werden — und die Dokumente
> müssen neu indiziert werden.

### LLM

Beliebig viele Modelle unter `models\`. Sie erscheinen automatisch im Dropdown;
im Code steht kein Modellname. Ordner, deren Name auf einen Embedder hindeutet
(`e5-`, `minilm`, `bge`, `embed`, …), werden ausgefiltert.

Der Worker erkennt selbst, welche Quantisierung vorliegt, und probiert in
dieser Reihenfolge:

- **WebGPU:** `q4f16` → `q4` → `fp16` → `q8` → `int8` → `uint8`
- **WASM:** `q4` → `uint8` → `q8` → `int8` → `q4f16`

Unterstützt werden beide Namensschemata:

```
model_<dtype>.onnx                          Single-File (LFM2)
decoder_model_merged_<dtype>.onnx  +  embed_tokens_<dtype>.onnx   Split (Gemma 4)
```

#### External Data — die häufigste Fehlerquelle

Überschreitet ein ONNX-Graph 2 GB, erlaubt das Protobuf-Format keine einzelne
Datei mehr. Die Gewichte werden dann ausgelagert, und die `.onnx` ist nur noch
ein Stub von 200–300 kB:

```
model_q4f16.onnx           222 kB    ← nur der Graph, allein nutzlos
model_q4f16.onnx_data     1.06 GB
model_q4f16.onnx_data_1    470 MB
```

**Alle Shards müssen mit.** Fehlt einer, lehnt der Worker die Variante ab und
nennt die fehlende Datei — statt mitten im Laden abzustürzen.

#### Grössenobergrenze

ONNX Runtime Web läuft in 32-Bit-WASM: der adressierbare Heap endet bei 4 GiB,
unabhängig vom System-RAM. Gewichte werden beim Laden durch diesen Heap
gestaget, auch wenn sie danach auf der GPU landen. Dazu kommt
`maxBufferSize` (typisch 2 GiB pro Buffer).

**Praktische Obergrenze: rund 3,4 GB Gesamtgewichte**, entsprechend etwa
4–6 Mrd. Parametern in q4. Der Worker prüft das vorab und lehnt zu grosse
Modelle mit klarer Meldung ab.

Modelle darüber — auch MoE-Modelle wie LFM2.5-8B-A1B — sind mit dieser Runtime
nicht lauffähig. **MoE reduziert den Rechenaufwand pro Token, nicht den
Speicherbedarf:** alle Experten müssen geladen sein, nur ein Bruchteil rechnet.

---

## Bedienung

- **Dokumente:** Icon links oder `+` im Eingabefeld. Einzeldateien über
  „Laden", ganze Ordner über „Ordner data\ einlesen" (bereits indizierte
  Dateien werden anhand des Dateinamens übersprungen).
- **Modell wählen:** Dropdown oben rechts. „Modell laden" lädt es vorab,
  damit die erste Frage nicht wartet.
- **Fragen:** unten eintippen, Enter.
- **Modus:** Ohne indizierte Dokumente antwortet tiseR als reiner Chat.
  Mit Dokumenten läuft RAG — sichtbar am Hinweis unter dem Eingabefeld.

### Unterstützte Formate

PDF · DOCX · XLSX · PPTX · CSV · TXT · EML · MSG · MD

Office-Originale liefern messbar besseres Retrieval als PDF-Exporte derselben
Datei — insbesondere bei PowerPoint. Wo das Original verfügbar ist, sollte es
verwendet werden.

### Suchsyntax

Frei mit normalen Fragen kombinierbar. Ohne Operator verhält sich alles wie
eine gewöhnliche Frage.

| Ausdruck | Wirkung |
|---|---|
| `tags:frist` | nur Chunks mit diesem Tag (`frist`, `schwellenwert`, `zustaendigkeit`) |
| `source:BöB` | nur Dokumente, deren Name den Begriff enthält |
| `filename:*.pdf` | Dateiname nach Glob-Muster |
| `"genaue Phrase"` | muss wörtlich vorkommen |
| `/CHF \d+/` | regulärer Ausdruck im Text |
| `-Intranet` bzw. `NOT Intranet` | Wort ausschliessen |

---

## Wie die Suche funktioniert

1. **Chunking** (`app.py`): strukturbewusst. Überschriften werden erkannt und
   als Präfix in den Chunk übernommen, Seiten- und Foliengrenzen sind harte
   Schnittpunkte, Inhaltsverzeichnis-Punktlinien und wiederkehrende Fusszeilen
   fliegen raus. Zielgrösse ~500 Zeichen.
2. **Metadaten-Tags**: per Regex erkannte Inhaltstypen (Frist, Schwellenwert,
   Zuständigkeit) werden **getrennt vom Text** gespeichert und gelangen nie ins
   Embedding-Fenster.
3. **Embedding**: im Browser, `passage: `-Präfix für Chunks, `query: ` für
   Suchanfragen. Beide Seiten müssen zum Trainingsschema des Modells passen.
4. **Retrieval**: Reciprocal Rank Fusion aus Cosinus-Ähnlichkeit und BM25.
   Strukturierte Operatoren wirken als harter Vorfilter davor.

`TAG_BOOST` in `app.py` ist verdrahtet, steht aber auf `1.0` (= aus). Er wird
erst scharfgeschaltet, wenn `eval_harness.py` den Effekt gegen reines BM25
gemessen hat. Eine Zahl ohne Messung wäre Bauchgefühl.

---

## Messen statt schätzen

```
python eval_harness.py
```

Misst Retrieval-Recall@5 gegen das Golden-Set in der Datei. **Nur Retrieval** —
ob das LLM die gefundene Stelle korrekt wiedergibt, ist eine separate Frage.

Zwei Fehlerklassen, die sich nicht verwechseln lassen dürfen:

- **Retrieval-Fehler**: der richtige Chunk kommt gar nicht im Kontext an.
  Kein grösseres LLM repariert das.
- **Dekodier-Fehler**: der Chunk war da, die Antwort trotzdem falsch.
  Beobachtet: „ISO 901" statt „9001", „1320" statt „>1320" — aus einer
  Näherung wird eine falsche Präzision.

`repetition_penalty` muss deshalb **exakt 1.0** bleiben. Jeder Wert darüber
bestraft wiederholte Tokens — und Ziffern wiederholen sich in Zahlen.

Im Browser-Log stehen pro Antwort Backend, dtype, TTFT und tok/s:

```
[tiseR] AKTIV: WEBGPU · q4f16 · LFM2.5-2.6B
[tiseR] webgpu/q4f16 — TTFT 1.42 s · 18.30 tok/s · gesamt 9.6 s
```

Steht dort `wasm`, läuft das Modell auf der CPU — typisch Faktor 5–10 langsamer.
Der Fallback ist absichtlich still, deshalb dieser Nachweis.

---

## Fehlersuche (Edge-Konsole mit F12)

| Symptom | Ursache |
|---|---|
| `SecurityError … origin 'null'` | `index.html` doppelgeklickt statt Server benutzt |
| 404 auf `config.json`, Embedder lädt ewig | Ordnername ≠ `EMBED_MODEL` in `embed_worker.js` |
| `RangeError: Array buffer allocation failed` | Modell überschreitet den WASM-Heap |
| `file was not found locally at "…onnx_data_2"` | External-Data-Shards unvollständig |
| `Self-Test: Dim=0, erwartet 1024` | `tokenizer.json` oder `sentencepiece.bpe.model` fehlt |
| GPU-Auslastung nur ~25 % | normal — autoregressives Decodieren ist bandbreiten-, nicht rechengebunden |

---

## Bekannte offene Punkte

- **Umbenennung `armachat` → `tiseR` unvollständig.** Betroffen:
  `setup_shortcut.py` (verweist noch auf `C:\armachat`), Datenbankname
  `armachat.db`, Logdatei `armachat_error.log`, Logger-Name in `app.py`.
- `requirements.txt` listet transitive Abhängigkeiten noch nicht vollständig
  (`typing_extensions`, `colorama` traten beim Offline-Install auf).
- `setup_shortcut.py` nutzt PowerShell mit COM — unter AppLocker blockiert.
  Die Verknüpfung muss manuell erstellt werden (siehe Installation).
- Verfügbarkeit von SQLite FTS5 im ausgelieferten Python-Build ist ungeprüft.
- `eval_harness.py` braucht eine Prüfung numerischer Treue („1320" vs. „>1320")
  und einheitliche Dekodierparameter, bevor Ergebnisse belastbar sind.

---

## Update der Dokumente

Neue Dateien nach `data\` legen und „Ordner data\ einlesen" erneut ausführen.
Für einen vollständigen Neuaufbau die `.db`-Datei löschen und neu einlesen.
Die Modelle bleiben davon unberührt.

Nach Änderungen am Chunking oder am Passage-Präfix ist eine Neuindexierung
zwingend — die gespeicherten Vektoren passen sonst nicht mehr zum Code.
Änderungen am **Query**-Präfix allein erfordern das nicht.
