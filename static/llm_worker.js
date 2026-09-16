/**
 * tiseR — LLM-Worker   [Version 20260810v01]
 *
 * NEU in 20260810v01 — SHARD-ERKENNUNG, STOP-TOKENS, PLAUSIBILITAET:
 *  (I) STUB-SCHWELLE ABGESCHAFFT. probeVariant() entschied anhand von
 *      STUB_THRESHOLD (1 MB), OB ueberhaupt nach .onnx_data-Shards gesucht wird.
 *      Gemma4-Bund hat einen 2.4-MB-Graph-Stub -> ueber der Schwelle -> die
 *      1.85 GB + 0.70 GB Shards wurden NIE gezaehlt. Ergebnis: "0.00 GB" in der
 *      Konsole, Modell ohne Gewichte geladen, Decoder kopiert den Prompt statt
 *      zu antworten. Jetzt wird IMMER nach Shards gesucht; ihre Existenz (nicht
 *      die Stub-Groesse) entscheidet ueber External Data.
 *  (J) PLAUSIBILITAETSPRUEFUNG. Ein Decoder unter MIN_DECODER_BYTES kann keine
 *      Gewichte enthalten. Statt ihn zu laden und Prompt-Echo zu produzieren,
 *      wird er mit klarer Begruendung abgelehnt. Fehlerhafte Exporte fallen
 *      damit beim Laden auf, nicht erst im Benchmark.
 *  (K) STOP-TOKENS AUS DEM TOKENIZER. Gemma 4 beendet Turns mit '<turn|>'
 *      (eot_token, ID 106), NICHT mit '<eos>'. Ohne passende eos_token_id laeuft
 *      der Decode bis max_new_tokens -> Timeout. eot_token/eos_token werden aus
 *      der tokenizer_config gelesen und als eos_token_id durchgereicht.
 *      Kein hardcodierter Modellname.
 *  (L) CLEANUP kennt '<turn|>' und '<|turn>'.
 *  (M) ALLOW_WASM_FALLBACK zurueck auf false (q2f16-Diagnose ist abgeschlossen).
 *
 * NEU in 20260805v04:
 *  (F) PROMPT_BUDGET 1800 -> 3000 und Kontext-Slice 3 -> 6. Das Backend lieferte
 *      TOP_K Chunks, der Worker warf davon die Haelfte weg. Bei 500-Zeichen-
 *      Chunks sah das Modell effektiv ~1.5 kB Beleg — zu wenig fuer Fragen,
 *      deren Antwort in einer Nebenzeile steht.
 *  (G) tok/s-MESSUNG KORRIGIERT. Der Nenner lief bis zum Aufloesen der
 *      Generator-Promise, enthielt also einen festen Overhead NACH dem letzten
 *      Token. Bei kurzen Antworten dominierte dieser Overhead und drueckte die
 *      Rate auf 0.5 tok/s, waehrend sie sich bei langen Antworten heraus-
 *      mittelte. Der Vergleich mass damit Antwortlaenge, nicht Geschwindigkeit.
 *      Jetzt endet die Messung beim LETZTEN Streamer-Callback.
 *  (H) TOKEN-BUDGET FUER REASONING-MODELLE. Bei THINKING=true wird das vom
 *      Frontend gelieferte Budget um THINK_BUDGET_EXTRA erhoeht. Der Denk-Trace
 *      zaehlt voll gegen das Limit; 650 reichten nachweislich nicht bis zur
 *      Antwort (Testfrage Power-BI-Kosten: abgeschnitten vor dem ersten Wort).
 *      Das Frontend muss dafuer NICHT wissen, ob ein Modell denkt.
 *
 * NEU in 20260805v03 — REASONING-MODELLE (LFM2.5 u.a.):
 *  (A) Das Chat-Template von LFM2.5 endet mit "<|im_start|>assistant\n<think>".
 *      Der OEFFNENDE Tag steht also im PROMPT, nicht in der Ausgabe. Der alte
 *      Cleanup-Regex /<think>[\s\S]*?<\/think>/ konnte deshalb NIE greifen —
 *      der komplette englische Denk-Trace landete ungefiltert in der Bubble.
 *      Trennlinie ist allein das SCHLIESSENDE '</think>'.
 *  (B) THINKING wird dynamisch aus generator.tokenizer.chat_template erkannt
 *      (kein hardcodierter Modellname). Nicht-Reasoning-Modelle laufen
 *      unveraendert wie bisher.
 *  (C) Streaming filtert live: Tokens VOR '</think>' werden zurueckgehalten und
 *      nie gepostet. Erst danach fliesst Text in die Bubble.
 *  (D) 'phase'-Nachrichten (prefill | thinking | answering) fuer die UI-Anzeige.
 *      Ohne requestId-Handler in aelteren index.html laufen sie folgenlos durch.
 *  (E) Reicht das Token-Budget nicht bis '</think>', gibt es eine KLARE Meldung
 *      statt einer leeren oder abgeschnittenen Antwort.
 *
 * Basis: 20260805v02
 * Basis: 20260711v01 (lokal). Alle dortigen Faehigkeiten bleiben erhalten:
 *   - Single-File CausalLM     -> model_<dtype>.onnx                   (z. B. LFM2-2.6B-Bund)
 *   - Split ConditionalGen     -> decoder_model_merged_<dtype>.onnx    (z. B. Gemma 4 E4B)
 *                                 + embed_tokens_<dtype>.onnx
 *   - dtype PRO SESSION bei Split-Modellen
 *   - '_quantized' -> '_q8' Uebersetzung im fetch-Override
 *   - ALLOW_WASM_FALLBACK / FORCE_WASM als Diagnose-Schalter
 *
 * NEU in 20260805v02:
 *  (1) EXTERNAL DATA. Ueberschreitet ein ONNX-Graph 2 GB, muessen die Gewichte
 *      in separate model_<dtype>.onnx_data[_N] ausgelagert werden; die .onnx ist
 *      dann nur ein Stub von ~200-300 kB. Die alte HEAD-Probe meldete fuer so
 *      einen Stub "vorhanden" — auch wenn kein einziger Shard danebenlag. Ergebnis:
 *      Absturz mitten im Laden statt sauberer Ablehnung. probeVariant() prueft
 *      jetzt die GROESSE und zaehlt die Shards luecklos ab .onnx_data, .onnx_data_1, ...
 *  (2) BUDGET-PRUEFUNG VOR DEM LADEN. ONNX Runtime Web laeuft in 32-Bit-WASM:
 *      der adressierbare Heap endet bei 4 GiB, unabhaengig vom System-RAM.
 *      Gewichte werden beim Laden durch diesen Heap gestaged, auch wenn sie
 *      danach auf der GPU landen. Zu grosse Kandidaten werden mit KLARER Meldung
 *      verworfen statt mit "RangeError: Array buffer allocation failed".
 *      Bei Split-Modellen zaehlt die SUMME beider Sessions.
 *  (3) BACKEND-NACHWEIS. ACTIVE_DEVICE/ACTIVE_DTYPE werden als eigene
 *      'backend'-Nachricht gepostet und in der Konsole als Banner ausgegeben
 *      (bei WASM als console.warn). Zusaetzlich TTFT und tok/s pro Antwort als
 *      'perf'-Nachricht. Damit ist "laeuft es wirklich auf der GPU?" messbar
 *      statt Glaubenssache. index.html braucht dafuer KEINE Aenderung: beide
 *      Nachrichten haben keine requestId und laufen im bestehenden onmessage
 *      folgenlos durch.
 *
 * Ablegen unter:  C:\tiseR\static\llm_worker.js
 */
let MODEL_NAME  = '';          // kommt IMMER per 'load' aus index.html (Dropdown)
const ORIGIN    = self.location.origin;
const TJS_URL   = ORIGIN + '/static/transformersjs-420/transformers.min.js';
const WASM_BASE = ORIGIN + '/static/transformersjs-420/';
let MODEL_BASE  = ORIGIN + '/models/' + MODEL_NAME;
let PROMPT_BUDGET = 3000;      // 20260805v04: angehoben (war 1800) — mehr Beleg pro Antwort
const THINK_BUDGET_EXTRA = 700; // Aufschlag auf max_new_tokens bei Reasoning-Modellen

// dtype-Kandidaten in Praeferenzreihenfolge. Erstes VOLLSTAENDIGES und ins
// Budget passendes File gewinnt.
// q4 vor fp16: fp16 ist bei 2.6B rund 5.4 GB und scheitert ohnehin am Budget —
// die Reihenfolge soll dort nicht erst haengenbleiben.
const DTYPE_WEBGPU = ['q4f16', 'q4', 'fp16', 'q8', 'int8', 'uint8', 'q2f16'];
const DTYPE_WASM   = ['q4', 'uint8', 'q8', 'int8', 'q4f16', 'q2f16'];
// q2f16 steht BEWUSST an letzter Stelle: 2-Bit-Blockquantisierung wird von
// ONNX Runtime Web derzeit nicht unterstuetzt (GatherBlockQuantized akzeptiert
// nur bits==4 oder bits==8). Der Eintrag existiert, damit solche Modelle
// ueberhaupt erkannt werden und der ECHTE Fehler aus der Session kommt statt
// eines irrefuehrenden 'Datei fehlt'. Er darf nie ein 4-/8-Bit-Modell verdraengen.

// Stiller WASM-Fallback: laedt bei einem WebGPU-Fehler das GESAMTE Modell noch
// einmal und ueberdeckt dabei die urspruengliche Fehlermeldung.
// Fuer die Fehlersuche auf false. Fuer den Produktivbetrieb ggf. wieder true.
const ALLOW_WASM_FALLBACK = false;  // 20260810v01: Diagnose beendet -> echte Fehler sichtbar lassen.

// DIAGNOSE: WebGPU erzwungen deaktivieren, um WebGPU-vs-CPU zu isolieren.
// true = laeuft auf WASM/CPU (langsam!), false = normal (WebGPU wenn verfuegbar).
const FORCE_WASM = false;

// --- Groessengrenzen (Bytes) ----------------------------------------------
const HEAP_LIMIT     = 4 * 1024 * 1024 * 1024;      // 32-Bit-WASM-Adressraum
const TOTAL_BUDGET   = 5.5 * 1024 * 1024 * 1024;    // Reserve fuer KV-Cache/Runtime
const MAX_SHARDS     = 64;                          // Schutz gegen Endlosschleife
// Plausibilitaet: ein Decoder-Graph ohne Gewichte ist wenige MB gross. Alles
// unterhalb dieser Grenze KANN kein lauffaehiges Sprachmodell sein — laden wuerde
// nur Prompt-Echo erzeugen. Bewusst grosszuegig: selbst ein 0.5B-Modell in q4
// liegt deutlich darueber.
const MIN_DECODER_BYTES = 100 * 1024 * 1024;        // 100 MB

let generator = null, loading = false, generating = false;
// Wird beim Laden aus dem Chat-Template abgeleitet (siehe loadModel).
let THINKING = false;
const THINK_CLOSE = '</think>';
let ACTIVE_DEVICE = null, ACTIVE_DTYPE = null;
let STOP_IDS = null;        // aus dem Tokenizer abgeleitete eos_token_id-Liste
let STOP_STRINGS = [];      // dieselben Marker als Text, fuer den Cleanup
let GPU_LIMITS = null;
const queue = [];

const SYSTEM_PROMPT =
  'Du bist ein Assistent für interne Prozessdokumentation. Antworte auf Deutsch, ' +
  'direkt und knapp. Beginne SOFORT mit der Antwort auf die Frage. Wiederhole NICHT ' +
  'den Dokumenttext, den Dateinamen, den Titel oder die Quelle. Antworte nur auf Basis ' +
  'der bereitgestellten Dokumente; du darfst mehrere Stellen kombinieren. Enthalten die ' +
  'Dokumente keine Grundlage, sage das in einem Satz. Erfinde keine Fakten. ' +
  'Keine Emojis, keine Markdown-Formatierung.';

// fetch()-Override: leitet Modell-Dateien nach /models/<name>/ um.
const _origFetch = self.fetch.bind(self);
self.fetch = function (url, opts) {
  let u = (typeof url === 'string') ? url : (url && url.url) ? url.url : String(url);
  if (u.indexOf(ORIGIN) >= 0 || u.endsWith('.wasm') || u.endsWith('.mjs')) return _origFetch(url, opts);
  const fname = u.split('?')[0].split('/').pop();
  // .onnx UND .onnx_data / .onnx_data_1 (External-Data-Shards) -> onnx\-Unterordner
  if (u.indexOf('.onnx') >= 0) {
    // Transformers.js uebersetzt dtype 'q8' in das Suffix '_quantized'.
    // Unsere Dateien heissen '_q8' -> hier zurueckuebersetzen. Gilt auch fuer
    // die Datenshards (_quantized.onnx_data_3 -> _q8.onnx_data_3).
    const fixed = fname.replace('_quantized.onnx', '_q8.onnx');
    return _origFetch(MODEL_BASE + '/onnx/' + fixed, opts);
  }
  const known = ['tokenizer.json','tokenizer_config.json','config.json',
                 'generation_config.json','special_tokens_map.json','vocab.json','merges.txt','added_tokens.json'];
  for (const k of known) if (fname === k || u.indexOf(k) >= 0) return _origFetch(MODEL_BASE + '/' + k, opts);
  if (u.indexOf('huggingface.co') >= 0 || u.indexOf('hf.co') >= 0) return _origFetch(MODEL_BASE + '/' + fname, opts);
  return _origFetch(url, opts);
};

self.onmessage = async (e) => {
  const d = e.data || {};
  if (d.type === 'load') {
    if (d.model_name) { MODEL_NAME = d.model_name; MODEL_BASE = ORIGIN + '/models/' + MODEL_NAME; }
    await loadModel();
  } else if (d.type === 'generate') {
    if (!generator) await loadModel();
    if (!generator) return;
    queue.push(d);
    drain();
  } else if (d.type === 'backend_query') {
    self.postMessage({ type: 'backend', device: ACTIVE_DEVICE, dtype: ACTIVE_DTYPE, limits: GPU_LIMITS });
  }
};

async function drain() {
  if (generating || !queue.length) return;
  generating = true;
  while (queue.length) { const j = queue.shift(); await generate(j); }
  generating = false;
}

let TextStreamer = null;   // wird beim Laden aus transformers.js geholt

// --------------------------------------------------------------- dtype-Erkennung
const fmtGB = b => (b / 1024 / 1024 / 1024).toFixed(2) + ' GB';

/** HEAD-Probe: Groesse in Bytes, 0 wenn vorhanden aber ohne Content-Length,
 *  -1 wenn nicht vorhanden. */
async function probeSize(url) {
  try {
    const r = await _origFetch(url, { method: 'HEAD' });
    if (!r || !r.ok) return -1;
    const len = r.headers.get('content-length');
    return len ? parseInt(len, 10) : 0;
  } catch (_) { return -1; }
}

/** Prueft EINEN Praefix+dtype auf Vollstaendigkeit.
 *  Rueckgabe {ok, total, largest, shards} oder {ok:false, reason}. */
async function probeVariant(prefix, dt) {
  const base = MODEL_BASE + '/onnx/' + prefix + dt;
  const stub = await probeSize(base + '.onnx');
  if (stub < 0) return { ok: false, reason: prefix + dt + '.onnx fehlt' };

  // IMMER nach External-Data-Shards suchen. Frueher entschied die Stub-Groesse
  // darueber, ob ueberhaupt gesucht wird — ein 2.4-MB-Stub galt damit faelsch-
  // licherweise als selbstenthalten und seine GB-Shards wurden ignoriert.
  // Die Existenz von '.onnx_data' ist das einzige verlaessliche Kriterium.
  let total = stub, largest = stub, shards = 1, external = false;
  const first = await probeSize(base + '.onnx_data');
  if (first >= 0) {
    external = true;
    total += first; shards++;
    if (first > largest) largest = first;
    for (let i = 1; i < MAX_SHARDS; i++) {
      const sz = await probeSize(base + '.onnx_data_' + i);
      if (sz < 0) break;                     // erste Luecke = Ende der Kette
      total += sz; shards++;
      if (sz > largest) largest = sz;
    }
  }
  return { ok: true, total, largest, shards, external, stub };
}

/** Waehlt fuer EINEN Praefix den besten vollstaendigen, ins Budget passenden dtype.
 *  budgetLeft begrenzt die Summe ueber mehrere Sessions (Split-Modelle). */
async function pickDtype(prefix, device, budgetLeft, rejected) {
  const list = (device === 'webgpu') ? DTYPE_WEBGPU : DTYPE_WASM;
  for (const dt of list) {
    const v = await probeVariant(prefix, dt);
    if (!v.ok) { rejected.push(v.reason); continue; }
    // Plausibilitaet: ein Decoder ohne Gewichte laedt zwar, antwortet aber nur
    // mit einer Kopie des Prompts. Lieber hier hart ablehnen.
    if (prefix.indexOf('decoder') >= 0 || prefix === 'model_') {
      if (v.total < MIN_DECODER_BYTES) {
        rejected.push(prefix + dt + ': nur ' + fmtGB(v.total) + ' gesamt' +
          (v.external ? ' (' + v.shards + ' Datei(en))'
                      : ' und KEINE .onnx_data-Shards daneben') +
          ' — das ist ein Graph ohne Gewichte. Ein solches Modell laedt, ' +
          'kopiert aber nur den Prompt. Export unvollstaendig.');
        continue;
      }
    }
    if (v.total > budgetLeft) {
      rejected.push(prefix + dt + ': ' + fmtGB(v.total) + ' ueberschreitet das ' +
        'verbleibende Budget von ' + fmtGB(budgetLeft) +
        ' (ONNX Runtime Web: 32-Bit-Heap, Limit ' + fmtGB(HEAP_LIMIT) + ')');
      continue;
    }
    if (device === 'webgpu' && GPU_LIMITS && GPU_LIMITS.maxBufferSize &&
        v.largest > GPU_LIMITS.maxBufferSize) {
      rejected.push(prefix + dt + ': groesster Shard ' + fmtGB(v.largest) +
        ' > maxBufferSize ' + fmtGB(GPU_LIMITS.maxBufferSize));
      continue;
    }
    console.log('[tiseR] ' + prefix + dt + ' — ' + v.shards + ' Datei(en), ' +
                fmtGB(v.total) + ', groesster Shard ' + fmtGB(v.largest));
    return { dt, v };
  }
  return null;
}

/** Liefert {dtype, total, rejected}.
 *  dtype ist ein String (Single-File) oder {session: dtype} (Split), oder null. */
async function detectDtypeMap(device) {
  const rejected = [];

  // Split-Modell zuerst pruefen (spezifischerer Praefix).
  const dec = await pickDtype('decoder_model_merged_', device, TOTAL_BUDGET, rejected);
  if (dec) {
    const emb = await pickDtype('embed_tokens_', device, TOTAL_BUDGET - dec.v.total, rejected);
    if (!emb) throw new Error(
      'decoder_model_merged_' + dec.dt + '.onnx ist vollstaendig, aber kein passendes ' +
      'embed_tokens_<dtype>.onnx daneben. Split-Modell unvollstaendig oder zu gross.\n' +
      rejected.join('\n'));
    return {
      dtype: { embed_tokens: emb.dt, decoder_model_merged: dec.dt },
      total: dec.v.total + emb.v.total, rejected
    };
  }

  const single = await pickDtype('model_', device, TOTAL_BUDGET, rejected);
  if (single) return { dtype: single.dt, total: single.v.total, rejected };
  return { dtype: null, total: 0, rejected };
}

function dtypeLabel(dt) {
  return (typeof dt === 'string') ? dt : JSON.stringify(dt);
}

// --------------------------------------------------------------- Laden
async function loadModel() {
  if (generator || loading) return;
  loading = true;
  post('status', 'Transformers.js lädt…', 0);
  try {
    const tjs = await import(TJS_URL);
    const pipeline = tjs.pipeline || (tjs.default && tjs.default.pipeline);
    const env      = tjs.env      || (tjs.default && tjs.default.env);
    TextStreamer   = tjs.TextStreamer || (tjs.default && tjs.default.TextStreamer) || null;
    if (!pipeline || !env) throw new Error('pipeline/env nicht gefunden');

    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.useBrowserCache = false;
    env.localModelPath = '/models/';
    if (env.backends?.onnx?.wasm) {
      env.backends.onnx.wasm.wasmPaths = WASM_BASE;
      env.backends.onnx.wasm.numThreads = 1;
      env.backends.onnx.wasm.proxy = false;
    }

    // WebGPU verfuegbar? Limits merken — sie begrenzen die Modellwahl.
    let device = 'wasm';
    if (!FORCE_WASM && self.navigator && self.navigator.gpu) {
      try {
        const adapter = await self.navigator.gpu.requestAdapter();
        if (adapter) {
          device = 'webgpu';
          const L = adapter.limits || {};
          GPU_LIMITS = {
            maxBufferSize: L.maxBufferSize || 0,
            maxStorageBufferBindingSize: L.maxStorageBufferBindingSize || 0,
            maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize || 0
          };
          console.log('=== WebGPU-Limits ===');
          console.log('maxBufferSize:               ', L.maxBufferSize, '(' + fmtGB(L.maxBufferSize) + ')');
          console.log('maxStorageBufferBindingSize: ', L.maxStorageBufferBindingSize, '(' + fmtGB(L.maxStorageBufferBindingSize) + ')');
          console.log('maxComputeWorkgroupStorageSize:', L.maxComputeWorkgroupStorageSize);
        }
      } catch (e) { console.warn('requestAdapter fehlgeschlagen:', e); }
    }
    if (FORCE_WASM) console.warn('[tiseR] FORCE_WASM=true — WebGPU absichtlich uebersprungen.');

    // dtype(s) aus den TATSAECHLICH vorhandenen und passenden Dateien ableiten.
    const sel = await detectDtypeMap(device);
    if (!sel.dtype) throw new Error(
      'Kein vollstaendiges, passendes Modell in /models/' + MODEL_NAME + '/onnx/.\n' +
      sel.rejected.join('\n') +
      '\nHinweis: Bei External-Data-Modellen muessen <praefix>_<dtype>.onnx UND ' +
      'alle <praefix>_<dtype>.onnx_data[_N] im selben Ordner liegen.');

    const dtype = sel.dtype;
    post('status', 'LLM lädt (' + device.toUpperCase() + ', ' + dtypeLabel(dtype) +
                   ', ' + fmtGB(sel.total) + ')…', 5);

    let usedDevice = device, usedDtype = dtype;
    try {
      generator = await pipeline('text-generation', MODEL_NAME, {
        dtype, device,
        progress_callback: (p) => post('status', 'LLM lädt ' + (p.progress ? Math.round(p.progress) : 0) + '%…', p.progress || 0)
      });
    } catch (err) {
      // DIAGNOSE: den ECHTEN Grund sichtbar machen, bevor irgendein Fallback greift.
      console.error('=== LADEFEHLER (' + device + ') ===');
      console.error('name:   ', err && err.name);
      console.error('message:', err && err.message);
      console.error('stack:  ', err && err.stack);
      console.error('object: ', err);

      if (ALLOW_WASM_FALLBACK && device === 'webgpu') {
        const alt = await detectDtypeMap('wasm');
        if (!alt.dtype) throw err;
        post('status', 'GPU n/a — lade auf CPU (' + dtypeLabel(alt.dtype) + ')…', 5);
        generator = await pipeline('text-generation', MODEL_NAME, {
          dtype: alt.dtype, device: 'wasm',
          progress_callback: (p) => post('status', 'CPU-Laden ' + (p.progress ? Math.round(p.progress) : 0) + '%…', p.progress || 0)
        });
        usedDevice = 'wasm'; usedDtype = alt.dtype;
      } else throw err;
    }

    if (typeof generator !== 'function') throw new Error('Generator ist keine Funktion');

    ACTIVE_DEVICE = usedDevice;
    ACTIVE_DTYPE  = dtypeLabel(usedDtype);

    // Reasoning-Modell? Erkennung AUS DEM TEMPLATE, nicht aus dem Modellnamen.
    // LFM2.5 haengt in add_generation_prompt ein '<think>' an -> das Modell
    // beginnt zwingend mit einem Denk-Trace und schliesst ihn mit '</think>'.
    try {
      const tk  = generator.tokenizer || {};
      const tpl = tk.chat_template ||
                  (tk._tokenizer_config && tk._tokenizer_config.chat_template) || '';
      THINKING = typeof tpl === 'string' && tpl.indexOf('<think>') >= 0;
    } catch (_) { THINKING = false; }
    console.log('[tiseR] Reasoning-Modell (Template enthaelt <think>): ' + THINKING);

    // Stop-Tokens AUS DEM TOKENIZER, nicht aus einer Modellnamen-Tabelle.
    // Gemma 4 schliesst jeden Turn mit eot_token '<turn|>' (ID 106); '<eos>'
    // erscheint dort nie. Fehlt diese ID in eos_token_id, laeuft der Decode bis
    // max_new_tokens durch -> Timeout statt Antwort.
    try {
      const tk  = generator.tokenizer || {};
      const cfg = tk._tokenizer_config || {};
      const names = [];
      for (const key of ['eot_token', 'eos_token']) {
        let v = cfg[key];
        if (v && typeof v === 'object') v = v.content;
        if (typeof v === 'string' && v && names.indexOf(v) < 0) names.push(v);
      }
      const msst = cfg.model_specific_special_tokens || {};
      if (typeof msst.eot_token === 'string' && names.indexOf(msst.eot_token) < 0)
        names.push(msst.eot_token);
      STOP_STRINGS = names.slice();
      const ids = [];
      for (const n of names) {
        let id = null;
        try { id = tk.convert_tokens_to_ids ? tk.convert_tokens_to_ids([n])[0] : null; }
        catch (_) { id = null; }
        if (typeof id === 'number' && id >= 0 && ids.indexOf(id) < 0) ids.push(id);
      }
      STOP_IDS = ids.length ? ids : null;
      console.log('[tiseR] Stop-Tokens: ' + (names.join(' ') || '(keine)') +
                  ' -> IDs ' + (STOP_IDS ? STOP_IDS.join(',') : '(aus generation_config)'));
      if (!STOP_IDS) console.warn('[tiseR] Keine Stop-Token-IDs ableitbar — es gilt ' +
        'allein die eos_token_id aus generation_config.json. Fehlt die Datei, ' +
        'laeuft jeder Decode bis max_new_tokens.');
    } catch (_) { STOP_IDS = null; STOP_STRINGS = []; }

    loading = false;

    // Harter Nachweis: unabhaengig von Status-Updates, unuebersehbar in der Konsole.
    const banner = '[tiseR] AKTIV: ' + ACTIVE_DEVICE.toUpperCase() + ' · ' +
                   ACTIVE_DTYPE + ' · ' + MODEL_NAME;
    if (ACTIVE_DEVICE === 'wasm') console.warn(banner + '   <-- CPU! Nicht die GPU.');
    else console.log(banner);
    self.postMessage({ type: 'backend', device: ACTIVE_DEVICE, dtype: ACTIVE_DTYPE, limits: GPU_LIMITS });
    self.postMessage({ type: 'ready' });
  } catch (err) {
    loading = false;
    self.postMessage({ type: 'error', text: err.message || String(err) });
  }
}

async function generate(job) {
  const baseLimit  = job.maxNewTokens > 0 ? job.maxNewTokens : 400;
  // Denk-Trace zaehlt voll gegen das Budget -> bei Reasoning-Modellen aufstocken.
  const tokenLimit = THINKING ? baseLimit + THINK_BUDGET_EXTRA : baseLimit;
  const requestId = job.requestId;
  const t0 = Date.now();
  let firstTokenAt = 0, lastTokenAt = 0, tokenCount = 0;
  try {
    self.postMessage({ type: 'generating', requestId });

    // Kontext proportional auf das Budget kürzen (Top-6 aus e5-large-Retrieval).
    let ctxText = '';
    const ctx = (job.context || []).slice(0, 6);
    if (ctx.length) {
      const totalRaw = ctx.reduce((s, c) => s + (c.content || '').length, 0);
      const perChunk = totalRaw <= PROMPT_BUDGET ? totalRaw : Math.max(200, Math.floor(PROMPT_BUDGET / ctx.length));
      ctxText = ctx.map(c => '[' + (c.source || 'DOK') + ']\n' + (c.content || '').slice(0, perChunk)).join('\n\n---\n\n');
    }

    const sysPrompt = job.systemPrompt || SYSTEM_PROMPT;

    const rawHist = Array.isArray(job.history) ? job.history : [];
    const maxTurns = ctxText ? 2 : 4;
    const perHistChars = ctxText ? 220 : 400;
    const histMsgs = rawHist.slice(-maxTurns).map(h => ({
      role: (h.role === 'assistant' ? 'assistant' : 'user'),
      content: String(h.content || '').slice(0, perHistChars)
    }));

    const currentUser = ctxText
      ? { role: 'user', content: 'Dokumente:\n\n' + ctxText + '\n\n---\nFrage: ' + job.prompt }
      : { role: 'user', content: job.prompt };
    const messages = [{ role: 'system', content: sysPrompt }, ...histMsgs, currentUser];

    self.postMessage({ type: 'phase', phase: 'prefill', requestId });

    // DIAGNOSE: der tatsaechlich gesendete Prompt-String. Zeigt sofort, ob das
    // Chat-Template greift (<|turn>user ... <|turn>model) und ob der Prompt mit
    // der AKTUELLEN Frage endet. Ohne diese Zeile ist jede Fehlersuche Raten.
    try {
      const dbg = generator.tokenizer.apply_chat_template(messages, {
        tokenize: false, add_generation_prompt: true });
      console.log('[tiseR] PROMPT (' + dbg.length + ' Zeichen) >>>\n' +
                  JSON.stringify(dbg.slice(0, 400)) + '\n   ... ENDE >>>\n' +
                  JSON.stringify(dbg.slice(-260)));
      console.log('[tiseR] messages-Rollen:', messages.map(m => m.role).join(' | '));
    } catch (e) { console.warn('[tiseR] Prompt-Log fehlgeschlagen:', e.message); }

    // Token-Streaming mit Reasoning-Filter.
    // Bei THINKING wird ALLES vor dem ersten '</think>' zurueckgehalten und nie
    // gepostet — die Bubble bleibt im "Denken"-Zustand, bis die eigentliche
    // Antwort beginnt. Der Puffer 'pre' waechst nur waehrend der Denkphase.
    let sawClose = !THINKING, pre = '';
    let streamer = null;
    if (TextStreamer && generator.tokenizer) {
      try {
        streamer = new TextStreamer(generator.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (txt) => {
            if (!txt) return;
            if (!firstTokenAt) {
              firstTokenAt = Date.now();
              self.postMessage({ type: 'phase',
                                 phase: THINKING ? 'thinking' : 'answering', requestId });
            }
            tokenCount++; lastTokenAt = Date.now();
            if (!sawClose) {
              pre += txt;
              const i = pre.indexOf(THINK_CLOSE);
              if (i < 0) return;                     // noch mitten im Denken
              sawClose = true;
              self.postMessage({ type: 'phase', phase: 'answering', requestId });
              const rest = pre.slice(i + THINK_CLOSE.length).replace(/^\s+/, '');
              if (rest) self.postMessage({ type: 'token', text: rest, requestId });
              return;
            }
            self.postMessage({ type: 'token', text: txt, requestId });
          }
        });
      } catch (_) { streamer = null; }
    }

    const timeoutMs = Math.max(180000, tokenLimit * 400 + 30000);
    const genOpts = {
      max_new_tokens: tokenLimit, do_sample: false,      // greedy = faktentreuer
      repetition_penalty: 1.0,                           // >1.0 verschluckt Ziffern
    };
    // Abgeleitete Stop-Tokens ergaenzen die generation_config. Ohne sie endet
    // ein Gemma-4-Turn nie (er schliesst mit '<turn|>', nicht mit '<eos>').
    if (STOP_IDS && STOP_IDS.length) genOpts.eos_token_id = STOP_IDS;
    if (streamer) genOpts.streamer = streamer;
    const result = await withTimeout(generator(messages, genOpts), timeoutMs);

    // Antwort extrahieren (Chat-Template -> letzter assistant-Turn)
    let output = result[0] && result[0].generated_text, answer = '';
    if (Array.isArray(output)) {
      // NUR die neu erzeugten Turns betrachten. Die alte Fassung filterte ueber
      // den GESAMTEN zurueckgegebenen Verlauf (Eingabe + Ausgabe) und griff bei
      // leerer Generierung auf output[last] zurueck — das ist die EINGABE, also
      // die eigene Frage bzw. der Dokumentenblock. Genau daher kamen die
      // "Antworten", die den vorherigen Prompt wiederholten.
      const fresh = output.slice(messages.length);
      const m = fresh.filter(x => x.role === 'assistant').pop();
      answer = m ? (m.content || '') : '';
      if (!answer && fresh.length === 0) {
        console.warn('[tiseR] Modell hat keinen neuen Turn erzeugt ' +
                     '(Eingabe: ' + messages.length + ' Nachrichten, ' +
                     'Rueckgabe: ' + output.length + ').');
      }
    } else { answer = String(output || ''); }

    // Bereinigung. ZUERST der Reasoning-Schnitt: der oeffnende <think>-Tag steht
    // im Prompt, in der Ausgabe erscheint nur das schliessende. Alles davor ist
    // Denk-Trace (meist englisch) und darf den Nutzer nie erreichen.
    const ci = answer.lastIndexOf(THINK_CLOSE);
    if (ci >= 0) {
      answer = answer.slice(ci + THINK_CLOSE.length).trim();
    } else if (THINKING) {
      // Budget reichte nicht bis zum Ende der Denkphase -> ehrlich melden,
      // statt einen halben englischen Trace als "Antwort" auszugeben.
      answer = '';
    }
    answer = answer.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    // Turn-Marker abschneiden. '<turn|>' (Gemma 4) kam bisher nicht vor —
    // steht er im Text, folgt dahinter der halluzinierte naechste Turn.
    const CUTS = ['<|im_end|>', '</s>', '<end_of_turn>', '<turn|>', '<|turn>', '<eos>']
                 .concat(STOP_STRINGS);
    for (const c of CUTS) { const i = answer.indexOf(c); if (i >= 0) answer = answer.slice(0, i); }
    answer = answer.trim();
    answer = answer.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/^#{1,4}\s+/gm, '').trim();

    const noInfo = /\s*[-–•]?\s*(?:dazu\s+)?enthalten\s+die\s+dokumente\s+keine\s+(?:konkrete\s+)?angabe[.…]*\s*$/i;
    if (noInfo.test(answer)) {
      const stripped = answer.replace(noInfo, '').trim();
      if (stripped.length >= 80) answer = stripped;
    }

    // Messwerte statt Bauchgefuehl. Der Unterschied WebGPU vs. WASM ist hier
    // sofort sichtbar (typisch Faktor 5-10 bei tok/s).
    const dur = (Date.now() - t0) / 1000;
    if (tokenCount > 1 && firstTokenAt) {
      const ttft = (firstTokenAt - t0) / 1000;
      // Nenner endet beim LETZTEN Token, nicht beim Aufloesen der Promise.
      const tps  = tokenCount / Math.max(0.001, ((lastTokenAt || Date.now()) - firstTokenAt) / 1000);
      console.log('[tiseR] ' + ACTIVE_DEVICE + '/' + ACTIVE_DTYPE +
                  ' — TTFT ' + ttft.toFixed(2) + ' s · ' + tps.toFixed(2) +
                  ' tok/s · gesamt ' + dur.toFixed(1) + ' s');
      self.postMessage({ type: 'perf', requestId, device: ACTIVE_DEVICE,
                         dtype: ACTIVE_DTYPE, ttft: +ttft.toFixed(2),
                         tps: +tps.toFixed(2), total: +dur.toFixed(1) });
    }

    const fallback = (THINKING && ci < 0)
      ? '(Antwort abgeschnitten: die Denkphase hat das Token-Budget von ' +
        tokenLimit + ' aufgebraucht, bevor die eigentliche Antwort begann. ' +
        'maxNewTokens erhoehen.)'
      : '(leere Antwort)';
    self.postMessage({ type: 'done', text: answer || fallback, requestId });
  } catch (err) {
    self.postMessage({ type: 'error', text: err.message || String(err), requestId });
  }
}

function withTimeout(p, ms) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('Timeout nach ' + (ms / 1000) + 's')), ms))]);
}
function post(type, text, progress) { self.postMessage({ type, text, progress }); }
