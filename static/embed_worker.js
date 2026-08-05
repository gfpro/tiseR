/**
 * tiseR — Embedding-Worker   [Version 20260805v01]  (multilingual-e5-large, 1024-dim, WASM)
 * Abgeleitet aus LoKI loki_embed_worker.js (bewährte Lade-Sequenz).
 *
 * Aenderungen ggue. 20260702v01:
 *  (1) QUERY_PREFIX korrigiert. Das eingesetzte Modell ist Xenova/multilingual-e5-large
 *      (base_model: intfloat/multilingual-e5-large) — NICHT die -instruct-Variante.
 *      Es wurde nie mit Instruktionstexten trainiert; die frueher verwendete
 *      Instruct-Vorlage war fuer das Modell reines Rauschen vor der Frage.
 *      Korrekt ist hier schlicht 'query: '.
 *      WICHTIG: Nur der QUERY-Pfad aendert sich. Die gespeicherten Passage-Vektoren
 *      bleiben gueltig — KEINE Neuindexierung noetig. Damit ist der Wechsel
 *      direkt A/B-messbar (siehe QUERY_MODE).
 *  (2) EMBED_MODEL auf den ehrlichen Ordnernamen 'multilingual-e5-large' gesetzt.
 *      Ein Ordner, der '-instruct' heisst und es nicht ist, produziert genau
 *      diese Verwechslung wieder.
 *  (3) DEADLOCK behoben. Kam waehrend des Ladens eine 'embed'-Anfrage, kehrte
 *      loadModel() wegen 'loading===true' sofort zurueck, extractor blieb null,
 *      und die Anfrage wurde STILL verworfen: keine Antwort, kein Fehler, die
 *      Promise in index.html loeste nie auf ("Embedding-Modell lädt…" fror ein).
 *      Jetzt wartet ein Nachzuegler auf die laufende Ladepromise; scheitert das
 *      Laden, wird die Anfrage mit ihrer requestId sichtbar abgelehnt.
 *  (4) Ladefehler nennt den Modellpfad. Ein 404 auf config.json (falscher
 *      Ordnername) war bisher nicht diagnostizierbar.
 *
 * Modelldateien erwartet unter:
 *   C:\tiseR\models\multilingual-e5-large\
 *     ├─ onnx\model_quantized.onnx   (int8, ~562 MB; dtype 'q8' fragt genau diese Datei an)
 *     ├─ config.json, tokenizer.json, tokenizer_config.json,
 *     ├─ special_tokens_map.json, sentencepiece.bpe.model
 *   Die Tokenizer-Dateien liegen im HAUPTordner, nur die .onnx in onnx\.
 *   model.onnx / model.onnx_data / model_fp16.onnx werden NICHT gebraucht.
 *
 * Ablegen unter:  C:\tiseR\static\embed_worker.js
 */
const EMBED_MODEL = 'multilingual-e5-large';
const ORIGIN     = self.location.origin;                 // portunabhängig
const TJS_URL    = ORIGIN + '/static/transformersjs-420/transformers.min.js';
const WASM_BASE  = ORIGIN + '/static/transformersjs-420/';
const MODEL_BASE = ORIGIN + '/models/' + EMBED_MODEL;
const EXPECT_DIM = 1024;   // e5-large: 1024 statt 384 bei MiniLM

// E5-Praefixe. Dokumente werden als 'passage: ' eingebettet, Suchanfragen als
// 'query: '. Beide Seiten MUESSEN zum Trainingsschema des Modells passen,
// sonst sinkt die Retrieval-Qualitaet messbar (kein optionaler Feinschliff).
const PASSAGE_PREFIX = 'passage: ';

// A/B-Schalter fuer den Eval-Harness. 'plain' ist fuer dieses Modell korrekt.
// 'instruct' reproduziert das alte (falsche) Verhalten, damit der Unterschied
// gegen dasselbe Golden-Set gemessen werden kann, statt geschaetzt zu werden.
// Nach der Messung diese Konstante auf 'plain' festnageln.
const QUERY_MODE = 'plain';   // 'plain' | 'instruct'
const QUERY_PREFIXES = {
  plain:    'query: ',
  instruct: 'Instruct: Retrieve semantically similar text.\nQuery: '
};
const QUERY_PREFIX = QUERY_PREFIXES[QUERY_MODE] || QUERY_PREFIXES.plain;

let extractor = null, loadPromise = null;

// fetch()-Override: leitet HuggingFace-Anfragen auf lokale Modelldateien um.
const _origFetch = self.fetch.bind(self);
self.fetch = function (url, opts) {
  let u = (typeof url === 'string') ? url : (url && url.url) ? url.url : String(url);
  if (u.indexOf(ORIGIN) >= 0 || u.endsWith('.wasm') || u.endsWith('.mjs')) return _origFetch(url, opts);
  const fname = u.split('?')[0].split('/').pop();
  if (u.indexOf('.onnx') >= 0) return _origFetch(MODEL_BASE + '/onnx/' + fname, opts);
  const known = ['tokenizer.json','tokenizer_config.json','config.json',
                 'special_tokens_map.json','sentencepiece.bpe.model',
                 'vocab.txt','vocab.json','merges.txt','added_tokens.json'];
  for (const k of known) if (fname === k || u.indexOf(k) >= 0) return _origFetch(MODEL_BASE + '/' + k, opts);
  if (u.indexOf('huggingface.co') >= 0 || u.indexOf('hf.co') >= 0) return _origFetch(MODEL_BASE + '/' + fname, opts);
  return _origFetch(url, opts);
};

self.onmessage = async (e) => {
  const d = e.data || {};
  if (d.type === 'load') {
    try { await loadModel(); } catch (_) { /* Fehler wurde bereits gepostet */ }
    return;
  }
  if (d.type === 'embed' || d.type === 'embed_query') {
    // Nachzuegler warten auf den laufenden Ladevorgang, statt still zu scheitern.
    try {
      await loadModel();
    } catch (err) {
      self.postMessage({ type: 'error', requestId: d.requestId,
                         text: 'Embedding-Modell nicht geladen: ' + (err.message || String(err)) });
      return;
    }
    if (d.type === 'embed') await embedItems(d.items, d.requestId);
    else                    await embedQuery(d.text, d.requestId);
  }
};

async function loadModel() {
  if (extractor) return extractor;
  if (loadPromise) return loadPromise;        // laufenden Ladevorgang abwarten
  loadPromise = (async () => {
    post('status', 'Embedding-Modell lädt…', 0);
    try {
      const tjs = await import(TJS_URL);
      const pipeline = tjs.pipeline || (tjs.default && tjs.default.pipeline);
      const env      = tjs.env      || (tjs.default && tjs.default.env);
      if (!pipeline || !env) throw new Error('pipeline/env nicht gefunden');

      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      env.useBrowserCache = false;
      env.localModelPath = '/models/';
      if (env.backends?.onnx?.wasm) {
        env.backends.onnx.wasm.wasmPaths = WASM_BASE;
        // numThreads>1 erfordert SharedArrayBuffer -> verlangt COOP/COEP-Header,
        // die app.py bereits setzt (Cross-Origin-Embedder-Policy: require-corp).
        // Damit laeuft das Query-Embedding (e5-large int8) mehrfaedig statt
        // single-thread -> spuerbar kuerzere Wartezeit VOR jeder RAG-Antwort.
        // Faellt SharedArrayBuffer aus (Header fehlen), still auf 1 zurueckfallen.
        env.backends.onnx.wasm.numThreads =
          (typeof SharedArrayBuffer !== 'undefined') ? Math.min(4, (self.navigator?.hardwareConcurrency || 4)) : 1;
        env.backends.onnx.wasm.proxy = false;
      }

      const ex = await pipeline('feature-extraction', EMBED_MODEL, {
        dtype: 'q8', device: 'wasm',
        progress_callback: (p) => post('status', 'Embedding-Modell ' + (p.progress ? Math.round(p.progress) : 0) + '%…', p.progress || 0)
      });

      // Self-Test: ein kaputter ORT/WASM-Build darf nicht als "ready" gelten.
      const t = Array.from((await ex(PASSAGE_PREFIX + 'test', { pooling: 'mean', normalize: true })).data);
      if (!t || t.length !== EXPECT_DIM) throw new Error('Self-Test: Dim=' + (t ? t.length : 0) + ', erwartet ' + EXPECT_DIM);

      extractor = ex;
      console.log('[tiseR] Embedder bereit: ' + EMBED_MODEL + ' · q8 · ' + EXPECT_DIM +
                  ' dim · QUERY_MODE=' + QUERY_MODE);
      self.postMessage({ type: 'ready' });
      return ex;
    } catch (err) {
      loadPromise = null;                     // erneuter Versuch bleibt moeglich
      const msg = (err.message || String(err)) +
        '  [Modellpfad: /models/' + EMBED_MODEL + '/ — bei 404 auf config.json ' +
        'stimmt der Ordnername nicht mit EMBED_MODEL ueberein]';
      console.error('[tiseR] Embedder-Ladefehler:', msg);
      self.postMessage({ type: 'error', text: msg });
      throw err;
    }
  })();
  return loadPromise;
}

async function embedItems(items, requestId) {
  const out = [];
  for (let i = 0; i < (items || []).length; i++) {
    const txt = (items[i].text || '').trim();
    if (!txt) continue;
    try {
      const v = await extractor(PASSAGE_PREFIX + txt, { pooling: 'mean', normalize: true });
      out.push({ id: items[i].id, vector: Array.from(v.data) });
    } catch (err) { console.warn('[embed] id=' + items[i].id, err.message); }
    post('status', 'Embedde ' + (i + 1) + '/' + items.length + '…', Math.round((i + 1) / items.length * 100));
  }
  self.postMessage({ type: 'embedded', items: out, requestId });
}

async function embedQuery(text, requestId) {
  try {
    const t = (text || '').slice(0, 400).trim();
    if (!t) throw new Error('Leerer Suchtext');
    const v = await extractor(QUERY_PREFIX + t, { pooling: 'mean', normalize: true });
    self.postMessage({ type: 'query_embedded', vector: Array.from(v.data), requestId });
  } catch (err) { self.postMessage({ type: 'error', text: err.message, requestId }); }
}

function post(type, text, progress) { self.postMessage({ type, text, progress }); }
