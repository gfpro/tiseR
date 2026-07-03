/**
 * tiseR — Embedding-Worker   [Version 20260702v01]  (multilingual-e5-large-instruct, 1024-dim, WASM)
 * Abgeleitet aus LoKI loki_embed_worker.js (bewährte Lade-Sequenz).
 *
 * WICHTIG bei E5-Modellen: Texte brauchen Praefixe, sonst geht der
 * Qualitaetsgewinn verloren. Dokument-Chunks -> 'passage: ', Suchanfragen
 * -> die Instruct-Form (siehe QUERY_PREFIX unten). Das passiert hier im
 * Worker zentral, damit app.py / index.html nichts davon wissen muessen.
 *
 * Modelldateien erwartet unter:
 *   C:\tiser\models\multilingual-e5-large-instruct\
 *     ├─ onnx\model_quantized.onnx   (int8, ~535 MB; dtype 'q8' fragt genau diese Datei an)
 *     ├─ tokenizer.json, tokenizer_config.json, config.json, special_tokens_map.json
 *   (Falls die .onnx anders heisst oder NICHT im onnx\-Unterordner liegt,
 *    den fetch-Override weiter unten anpassen.)
 *
 * Ablegen unter:  C:\tiser\static\embed_worker.js
 */
const EMBED_MODEL = 'multilingual-e5-large-instruct';
const ORIGIN     = self.location.origin;                 // portunabhängig
const TJS_URL    = ORIGIN + '/static/transformersjs-420/transformers.min.js';
const WASM_BASE  = ORIGIN + '/static/transformersjs-420/';
const MODEL_BASE = ORIGIN + '/models/' + EMBED_MODEL;
const EXPECT_DIM = 1024;   // e5-large: 1024 statt 384 bei MiniLM

// E5-Praefixe. Dokumente werden als 'passage: ' eingebettet, Suchanfragen mit
// der Instruct-Vorlage. Beide Seiten MUESSEN konsistent sein, sonst sinkt die
// Retrieval-Qualitaet messbar (das ist kein optionaler Feinschliff).
const PASSAGE_PREFIX = 'passage: ';
const QUERY_PREFIX   = 'Instruct: Retrieve semantically similar text.\nQuery: ';

let extractor = null, loading = false;

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
  if (d.type === 'load')        { await loadModel(); }
  else if (d.type === 'embed')        { if (!extractor) await loadModel(); if (extractor) await embedItems(d.items, d.requestId); }
  else if (d.type === 'embed_query')  { if (!extractor) await loadModel(); if (extractor) await embedQuery(d.text, d.requestId); }
};

async function loadModel() {
  if (extractor || loading) return;
  loading = true;
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
      // Faellt SharedArrayBuffer aus (Header fehlen), still auf 1 zuruckfallen.
      env.backends.onnx.wasm.numThreads =
        (typeof SharedArrayBuffer !== 'undefined') ? Math.min(4, (self.navigator?.hardwareConcurrency || 4)) : 1;
      env.backends.onnx.wasm.proxy = false;
    }

    extractor = await pipeline('feature-extraction', EMBED_MODEL, {
      dtype: 'q8', device: 'wasm',
      progress_callback: (p) => post('status', 'Embedding-Modell ' + (p.progress ? Math.round(p.progress) : 0) + '%…', p.progress || 0)
    });

    // Self-Test: ein kaputter ORT/WASM-Build darf nicht als "ready" gelten.
    const t = Array.from((await extractor(PASSAGE_PREFIX + 'test', { pooling: 'mean', normalize: true })).data);
    if (!t || t.length !== EXPECT_DIM) throw new Error('Self-Test: Dim=' + (t ? t.length : 0) + ', erwartet ' + EXPECT_DIM);

    loading = false;
    self.postMessage({ type: 'ready' });
  } catch (err) {
    loading = false;
    self.postMessage({ type: 'error', text: err.message || String(err) });
  }
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
