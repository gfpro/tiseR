# tiseR
Tiny Secure RAG

**Vollständig lokaler, offline-fähiger RAG-Chatbot.** Dokumente einlesen, semantisch
durchsuchen und vom LLM beantworten lassen — ohne Cloud, ohne Internet, ohne native
DLLs. Embedding und Sprachmodell laufen im Browser (Transformers.js / WebGPU mit
WASM-Fallback); das Python-Backend (Flask) übernimmt Extraktion, Chunking und
hybride Suche.

Entwickelt für eine restriktive Windows-Umgebung: **Python 3.10, keine
Adminrechte, AppLocker-Whitelist, kein Internet zur Laufzeit, keine `.pyd`/`.dll`.**
Alle Python-Abhängigkeiten sind reines Python.

---

## Inhalt

- [Funktionen](#funktionen)
- [Architektur](#architektur)
- [Voraussetzungen](#voraussetzungen)
- [Repository-Struktur](#repository-struktur)
- [Installation](#installation)
  - [1. Code ablegen](#1-code-ablegen)
  - [2. Python-Abhängigkeiten (offline)](#2-python-abhängigkeiten-offline)
  - [3. Transformers.js bereitstellen](#3-transformersjs-bereitstellen)
  - [4. Modelle bereitstellen](#4-modelle-bereitstellen)
  - [5. Starten](#5-starten)
  - [6. Desktop-Verknüpfung](#6-desktop-verknüpfung)
- [Dokumente für das RAG ablegen](#dokumente-für-das-rag-ablegen)
- [Bedienung](#bedienung)
- [Konfiguration](#konfiguration)
- [Troubleshooting](#troubleshooting)

---

## Funktionen

- **100 % lokal/offline** — keine externen Aufrufe zur Laufzeit.
- **Hybrid-Retrieval** — Pure-Python BM25 (numpy-frei) + Cosine-Similarity,
  fusioniert via Reciprocal Rank Fusion (RRF).
- **Strukturbewusstes Chunking** — Überschriften-Schnitte, ToC-/Kopf-/Fusszeilen
  werden entfernt, Seiten-/Foliengrenzen als harte Schnitte erhalten.
- **Viele Formate** — PDF, DOCX, XLSX, PPTX, CSV, TXT, EML, MSG.
  Office-Formate (DOCX/XLSX/PPTX) werden via stdlib `zipfile` + `ElementTree`
  geparst — **kein** `python-docx`/`openpyxl`/`lxml` (jeweils native DLLs).
- **Browser-seitige ML** — Embedding (`multilingual-e5-large-instruct`, 1024-dim,
  int8) und LLM laufen über Transformers.js im Web-Worker.
- **Persistenz** — Vektoren in SQLite, BM25-Index wird beim Start neu aufgebaut.
- **Chat- und RAG-Modus** — ohne Dokumente reiner Chat, mit Dokumenten RAG;
  Quellen werden pro Antwort ausklappbar angezeigt.

> **Hinweis zur Faktentreue:** Das LLM dekodiert greedy (`do_sample: false`) mit
> `repetition_penalty: 1.0`. Letzteres ist Absicht — jeder Wert > 1.0 lässt greedy
> wiederholte Ziffern bestrafen und korrumpiert Zahlen (z. B. „ISO 9001" → „ISO 901").

---

## Architektur

```
Browser (Edge)                         Python-Backend (Flask, app.py)
┌──────────────────────────┐           ┌──────────────────────────────────┐
│ index.html               │           │ Extraktion (pypdf, zipfile, …)   │
│  ├─ embed_worker.js  ────┼─ Vektoren─┼─▶ Chunking (strukturbewusst)     │
│  │   (e5-large, WASM)    │           │   Speicherung (SQLite)           │
│  └─ llm_worker.js        │  Suche ◀──┼── Hybrid: BM25 + Cosine (RRF)    │
│      (Gemma/LFM2,        │           │                                  │
│       WebGPU/WASM)       │  Kontext─▶│   /api/search liefert Top-K      │
└──────────────────────────┘           └──────────────────────────────────┘
```

Embedding und Generierung passieren **im Browser**, nicht in Python. Deshalb
enthält `requirements.txt` bewusst kein `torch`/`onnxruntime`/`faiss`.

---

## Voraussetzungen

- **Windows** mit **Python 3.10** (getestet; andere 3.x-Versionen vermutlich ok,
  aber ungetestet).
- **Microsoft Edge** (oder ein anderer Chromium-Browser mit WebGPU; ohne WebGPU
  greift automatisch der WASM-Fallback — langsamer, aber funktionsfähig).
- Genügend Platz für die Modelldateien (Embedder ~535 MB, LLM je nach Modell
  mehrere GB).
- Die Modelldateien und das Transformers.js-Bundle **sind nicht Teil dieses
  Repos** und müssen separat bereitgestellt werden (siehe unten).

---

## Repository-Struktur

```
tiseR/
├─ app.py                 # Flask-Backend: Extraktion, Chunking, Suche, API
├─ index.html             # Frontend (im Hauptordner, nicht in static/)
├─ requirements.txt       # Flask, pypdf, olefile  (alles reines Python)
├─ setup_shortcut.py      # erzeugt die Desktop-Verknüpfung lokal
├─ eval_harness.py        # misst Retrieval-Recall@k (optional, Entwicklung)
├─ README.md
│
├─ static/
│  ├─ embed_worker.js     # Embedding-Worker (e5-large)
│  ├─ llm_worker.js       # LLM-Worker (Gemma/LFM2)
│  └─ transformersjs-420/ # ⟵ MUSS bereitgestellt werden (Transformers.js + WASM)
│
├─ models/                # ⟵ MUSS bereitgestellt werden (Embedder + LLM[s])
│  ├─ multilingual-e5-large-instruct/
│  ├─ gemma-4-E2B-it/      (Beispiel)
│     └─ onnx/
│
└─ data/                  # ⟵ hier die RAG-Dokumente ablegen
```

> `static/transformersjs-420/`, `models/` und `data/` werden **nicht** mit
> ausgeliefert. Sie stehen in `.gitignore` (siehe unten) und müssen lokal befüllt
> werden.

---

## Installation

### 1. Code ablegen

Repo klonen oder ZIP entpacken nach `C:\tiser`:

```
C:\tiser\
├─ app.py
├─ index.html
├─ requirements.txt
└─ …
```

> Der Standard-Installationspfad in dieser Anleitung ist `C:\tiser`. Wenn du einen
> anderen Pfad nutzt, passe ihn in der Verknüpfung (Schritt 6) entsprechend an.

### 2. Python-Abhängigkeiten (offline)

`requirements.txt` enthält ausschliesslich reines Python (AppLocker-konform):

```
Flask==3.0.3
pypdf==5.9.0
olefile==0.47    # nur für .msg (binäres Outlook-OLE); optional
```

Auf einem Rechner **mit** Internet die Wheels herunterladen, auf den Zielrechner
übertragen und dort offline installieren:

```bat
:: Rechner mit Internet
pip download -r requirements.txt -d wheels

:: Wheels auf den Zielrechner kopieren, dann dort:
pip install --no-index --find-links wheels -r requirements.txt
```

`olefile` ist optional: Fehlt es, funktionieren alle anderen Formate weiter; nur
`.msg` meldet dann einen klaren Hinweis.

### 3. Transformers.js bereitstellen

Die Worker laden Transformers.js und die ONNX-Runtime-WASM-Backends lokal aus:

```
static/transformersjs-420/
├─ transformers.min.js     # von den Workern als Modul importiert
└─ … (ort-*.wasm, *.mjs)   # ONNX-Runtime-Web WASM-Backend
```

Die Worker erwarten genau diesen Ordnernamen (`transformersjs-420`, entspricht der
verwendeten Transformers.js-Version 4.2.x). Lege das vollständige Distributions-
Bundle inkl. der `.wasm`-/`.mjs`-Dateien dort ab. Die WASM-Pfade werden im Worker
auf diesen Ordner gesetzt (`numThreads: 1`, `proxy: false`).

> Wenn du eine andere Version verwenden willst, ändere `TJS_URL`/`WASM_BASE` in
> `static/embed_worker.js` **und** `static/llm_worker.js` und benenne den Ordner
> konsistent um.

### 4. Modelle bereitstellen

Alle Modelle liegen unter `models/`. **Pro Modell ein Ordner**, die `.onnx`-Dateien
im Unterordner `onnx/`, der Rest (Tokenizer, Config) direkt im Modellordner.

#### a) Embedding-Modell (Pflicht)

Ohne Embedder läuft nichts. Erwartet wird `multilingual-e5-large-instruct`
(1024-dim, int8):

```
models/multilingual-e5-large-instruct/
├─ onnx/
│  └─ model_quantized.onnx        # int8 / q8, ~535 MB  (dtype 'q8' fragt genau diese Datei an)
├─ tokenizer.json
├─ tokenizer_config.json
├─ config.json
├─ special_tokens_map.json
└─ sentencepiece.bpe.model         # falls vom Tokenizer benötigt
```

Heisst die `.onnx`-Datei anders oder liegt nicht im `onnx/`-Unterordner, passe den
`fetch`-Override / `dtype` oben in `static/embed_worker.js` an.

> E5-Modelle brauchen Präfixe (`passage:` für Dokumente, eine Instruct-Vorlage für
> Anfragen). Das macht der Worker zentral — das Backend muss davon nichts wissen.

#### b) Sprachmodell(e) (Pflicht, mindestens eines)

Beliebig viele LLM-Ordner unter `models/`. Jedes erscheint im Dropdown der
Oberfläche (Embedder werden anhand des Namensmusters automatisch ausgeschlossen).
Erwartete Dateien pro LLM:

```
models/<LLM-NAME>/
├─ onnx/
│  ├─ model_q4f16.onnx     # für WebGPU   (dtype 'q4f16')
│  └─ model_uint8.onnx     # CPU/WASM-Fallback (dtype 'uint8')
├─ tokenizer.json
├─ tokenizer_config.json
├─ config.json
├─ generation_config.json
├─ special_tokens_map.json
└─ (vocab.json, merges.txt, added_tokens.json — je nach Modell)
```

- **WebGPU vorhanden** → `model_q4f16.onnx` wird verwendet (schnell).
- **Kein WebGPU** → automatischer Fallback auf `model_uint8.onnx` (langsamer).

Stelle mindestens die zur Hardware passende Variante bereit. Beispiele für LLM-
Ordnernamen: `gemma-4-E2B-it` (klein/schnell), `gemma-4-E4B-it` (grösser/besser),
oder ein LFM2-Modell.

Modelle:
[Apertus-v1.1-4B-Instruct](https://huggingface.co/onnx-community/Apertus-v1.1-4B-Instruct-ONNX)
[Gemma 4 E2B](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)
[LFM2-2.6B](https://huggingface.co/onnx-community/LFM2-2.6B-ONNX)

> **Wichtig:** Lege **kein** Embedding-Modell ohne erkennbares Namensmuster in
> `models/` ab. Das Backend filtert Embedder am Namen (`e5-`, `minilm`, `embed`,
> `bge`, `gte-`, `nomic`, `sentence`). Ein Embedder, der diesem Muster nicht
> entspricht, taucht sonst fälschlich als „LLM" im Dropdown auf und produziert
> Kauderwelsch.

### 5. Starten

> ⚠️ **Häufigster Fehler:** `index.html` **nicht** doppelklicken. Unter `file://`
> kann der Browser keine Web-Worker starten („SecurityError … origin 'null'").
> **Immer** über den Server starten.

```bat
cd C:\tiser
python app.py
```

Dann in **Edge** öffnen: **http://localhost:8000**

Das Backend lauscht auf `127.0.0.1:8000`. Beim ersten Start entsteht die
Datenbankdatei im Anwendungsordner. Zum Beenden das Konsolenfenster schliessen.

### 6. Desktop-Verknüpfung

Ein einzelner Doppelklick auf eine `.lnk`-Verknüpfung startet den Server. Unter
AppLocker ist eine Verknüpfung, die **direkt auf `python.exe`** zeigt, die
zuverlässigste Einzelklick-Methode.

> ⚠️ **Lege keine fertige `.lnk` ins Repo.** Eine Windows-Verknüpfung bettet beim
> Erstellen rechnerspezifische Daten ein (Konto-SID, Rechnername, teils MAC). Eine
> committete `.lnk` veröffentlicht diese Fingerprints. Jeder Nutzer erzeugt die
> Verknüpfung **lokal** — entweder per Skript oder manuell.

**Variante A — per Skript:**

```bat
cd C:\tiser
python setup_shortcut.py
```

Das Skript ermittelt den echten Desktop-Pfad (funktioniert auch bei umgeleiteten
Desktops auf Netzlaufwerken) und legt dort die Verknüpfung an.

> Falls `setup_shortcut.py` noch auf einen alten Pfad/Namen zeigt, anpassen:
> ```python
> SCRIPT  = r"C:\tiser\app.py"
> WORKDIR = r"C:\tiser"
> # und Join-Path …  'tiseR.lnk'
> ```

**Variante B — manuell:**
Rechtsklick auf den Desktop → *Neu* → *Verknüpfung*, mit folgenden Werten:

| Feld               | Wert                                         |
|--------------------|----------------------------------------------|
| **Ziel**           | `C:\Program Files\Python310\python.exe`      |
| **Argumente**      | `"C:\tiser\app.py"`                           |
| **Ausführen in**   | `C:\tiser`                                    |
| **Name**           | `tiseR`                                       |

(Den Python-Pfad ggf. an die eigene Installation anpassen.)

Doppelklick auf die Verknüpfung startet den Server und öffnet den Browser.

---

## Dokumente für das RAG ablegen

Lege die zu durchsuchenden Dokumente in den Ordner **`data\`** im Anwendungsordner:

```
C:\tiser\data\
├─ handbuch.pdf
├─ prozesse.docx
└─ …
```

Unterstützte Formate: **PDF · DOCX · XLSX · PPTX · CSV · TXT · EML · MSG**.

Einlesen über die Oberfläche: Button **„Ordner data\ einlesen"**. Bereits
indizierte Dateien werden anhand des Dateinamens übersprungen. Einzelne Dateien
können alternativ direkt über **„Laden"** hochgeladen werden.

> **Empfehlung:** Office-Originale (DOCX/PPTX/XLSX) liefern saubere Struktur und
> Tabellen. PDF-Exporte derselben Inhalte verlieren oft Struktur — wenn das
> Original vorliegt, dieses bevorzugen.

**Dokumente aktualisieren:** Neue Dateien in `data\` legen und erneut einlesen. Für
einen kompletten Neuaufbau die Datenbankdatei löschen und neu einlesen — die
Modelle bleiben unberührt.

---

## Bedienung

- **Modell wählen:** Dropdown oben rechts. Modellwechsel lädt das Modell beim
  nächsten Senden neu. Optional **„Modell laden"** zum Vorab-Laden.
- **Fragen:** unten eintippen, *Enter* zum Senden (*Shift+Enter* = Zeilenumbruch).
- **Modus:** Ohne indizierte Dokumente → **Chat**. Mit Dokumenten → **RAG**
  (Antwort nur aus dem Kontext, Quellen ausklappbar).
- **Dokumente verwalten:** Dokumenten-Icon / **+** öffnet das Verwaltungsfenster
  (Liste mit Chunk-Anzahl, Einzel-Löschung, „Alle löschen").
- **Kurzzeitgedächtnis:** Die letzten Q&A-Turns fliessen in den Prompt ein.

---

## Konfiguration

Zentrale Stellschrauben in **`app.py`**:

| Konstante       | Bedeutung                                         |
|-----------------|---------------------------------------------------|
| `CHUNK_TARGET`  | Ziel-Zeichen pro Chunk (Standard 500)             |
| `CHUNK_HARD`    | harte Obergrenze pro Chunk (750)                  |
| `TOP_K`         | Anzahl Treffer pro Suche (6)                      |
| `RRF_K`         | RRF-Konstante (60)                                |
| `HOST` / `PORT` | Bind-Adresse (`127.0.0.1:8000`)                   |

In **`static/llm_worker.js`**: `PROMPT_BUDGET` (Zeichenbudget des Kontexts),
Default-Modellname und `dtype`-Varianten. In **`static/embed_worker.js`**:
`EMBED_MODEL`, `EXPECT_DIM`, E5-Präfixe.

Die System-Prompts für Chat- vs. RAG-Modus stehen oben in **`index.html`**
(`PROMPT_CHAT`, `PROMPT_RAG`).

---

## Troubleshooting

Edge-Konsole mit **F12** öffnen.

- **`SecurityError` / origin `'null'`** → `index.html` wurde doppelgeklickt statt
  über den Server geöffnet. Immer `python app.py` + `http://localhost:8000`.
- **404 auf eine `.onnx`-Datei** → im `models\<modell>\onnx\` fehlt die
  angeforderte dtype-Variante. Embedder: `model_quantized.onnx` (q8). LLM:
  `model_q4f16.onnx` (GPU) bzw. `model_uint8.onnx` (CPU).
- **Port belegt** → tiseR läuft vermutlich schon. Vorhandene Instanz im Browser
  öffnen oder die alte `python.exe` im Task-Manager beenden.
- **Embedder-Fehler beim Start** → Modellordner/-dateien prüfen (Pflicht: das
  e5-large-Embedding-Modell).
- **Antwort ist Kauderwelsch** → vermutlich ein Embedder als LLM ausgewählt; prüfen,
  dass nur echte Sprachmodelle ohne Embedder-Namensmuster im Dropdown stehen.


