/**
 * tiseR — LLM-Worker   [Version 20260708v02]
 * Modellunabhaengig: dtype wird aus den vorhandenen ONNX-Dateien ERKANNT
 * (HEAD-Probe), nicht am Modellnamen hartcodiert. Erkennt BEIDE Namensschemata:
 *   - Single-File CausalLM     -> model_<dtype>.onnx                   (z. B. LFM2-2.6B-Bund)
 *   - Split ConditionalGen     -> decoder_model_merged_<dtype>.onnx    (z. B. Gemma 4 E4B)
 * Beide laufen ueber dieselbe text-generation-Pipeline (wie in LoKI bewaehrt).
 * Vision/Audio-Encoder werden von der text-generation-Pipeline NICHT geladen.
 *
 * Ablegen unter:  C:\tiseR\static\llm_worker.js
 */
let MODEL_NAME  = '';          // kommt IMMER per 'load' aus index.html (Dropdown)
const ORIGIN    = self.location.origin;
const TJS_URL   = ORIGIN + '/static/transformersjs-420/transformers.min.js';
const WASM_BASE = ORIGIN + '/static/transformersjs-420/';
let MODEL_BASE  = ORIGIN + '/models/' + MODEL_NAME;
let PROMPT_BUDGET = 3000;      // Gesamt-Zeichenbudget fuer den Kontext (modellunabhaengig)

// dtype-Kandidaten in Praeferenzreihenfolge. Erstes vorhandenes File gewinnt.
const DTYPE_WEBGPU = ['q4f16', 'fp16', 'q4', 'q8', 'int8', 'uint8'];
const DTYPE_WASM   = ['q4', 'uint8', 'q8', 'int8', 'q4f16'];
// Dateinamen-Praefixe: Single-File ODER Gemma-Split.
const ONNX_PREFIXES = ['model_', 'decoder_model_merged_'];

let generator = null, loading = false, generating = false;
const queue = [];

const SYSTEM_PROMPT =
  'Du bist ein Assistent für interne Prozessdokumentation. Antworte auf Deutsch, ' +
  'nur auf Basis der bereitgestellten Dokumente. Du darfst Informationen aus ' +
  'mehreren Stellen kombinieren und zusammenfassen. Enthalten die Dokumente keine ' +
  'Grundlage für die Antwort, sage das offen. Erfinde keine Fakten. ' +
  'Keine Emojis, keine Markdown-Formatierung.';

// fetch()-Override: leitet Modell-Dateien nach /models/<name>/ um.
const _origFetch = self.fetch.bind(self);
self.fetch = function (url, opts) {
  let u = (typeof url === 'string') ? url : (url && url.url) ? url.url : String(url);
  if (u.indexOf(ORIGIN) >= 0 || u.endsWith('.wasm') || u.endsWith('.mjs')) return _origFetch(url, opts);
  const fname = u.split('?')[0].split('/').pop();
  // .onnx UND .onnx_data / .onnx_data_1 (Split-Dateien) -> onnx\-Unterordner
  if (u.indexOf('.onnx') >= 0) return _origFetch(MODEL_BASE + '/onnx/' + fname, opts);
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
  }
};

async function drain() {
  if (generating || !queue.length) return;
  generating = true;
  while (queue.length) { const j = queue.shift(); await generate(j); }
  generating = false;
}

let TextStreamer = null;   // wird beim Laden aus transformers.js geholt

// HEAD-Probe: welche <praefix><dtype>.onnx liegt im onnx\-Ordner? Erster Treffer
// gewinnt. Deckt model_q4.onnx (LFM2) UND decoder_model_merged_q4f16.onnx (Gemma).
async function detectDtype(device) {
  const list = (device === 'webgpu') ? DTYPE_WEBGPU : DTYPE_WASM;
  for (const dt of list) {
    for (const pre of ONNX_PREFIXES) {
      try {
        const r = await _origFetch(MODEL_BASE + '/onnx/' + pre + dt + '.onnx', { method: 'HEAD' });
        if (r && r.ok) return dt;
      } catch (_) { /* naechsten Kandidaten probieren */ }
    }
  }
  return null;
}

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

    // WebGPU verfuegbar?
    let device = 'wasm';
    if (self.navigator && self.navigator.gpu) {
      try { if (await self.navigator.gpu.requestAdapter()) device = 'webgpu'; } catch (_) {}
    }
    // dtype aus den TATSAECHLICH vorhandenen Dateien ableiten.
    let dtype = await detectDtype(device);
    if (!dtype) throw new Error(
      'Keine (decoder_model_merged_|model_)<dtype>.onnx in /models/' + MODEL_NAME + '/onnx/ gefunden (gesucht: ' +
      (device === 'webgpu' ? DTYPE_WEBGPU : DTYPE_WASM).join(', ') + ').');
    post('status', 'LLM lädt (' + device.toUpperCase() + ', ' + dtype + ')…', 5);

    try {
      generator = await pipeline('text-generation', MODEL_NAME, {
        dtype, device,
        progress_callback: (p) => post('status', 'LLM lädt ' + (p.progress ? Math.round(p.progress) : 0) + '%…', p.progress || 0)
      });
    } catch (err) {
      if (device === 'webgpu') {
        const wdt = await detectDtype('wasm');
        post('status', 'GPU n/a — lade auf CPU (' + (wdt || '?') + ')…', 5);
        if (!wdt) throw err;
        generator = await pipeline('text-generation', MODEL_NAME, {
          dtype: wdt, device: 'wasm',
          progress_callback: (p) => post('status', 'CPU-Laden ' + (p.progress ? Math.round(p.progress) : 0) + '%…', p.progress || 0)
        });
      } else throw err;
    }

    if (typeof generator !== 'function') throw new Error('Generator ist keine Funktion');
    loading = false;
    self.postMessage({ type: 'ready' });
  } catch (err) {
    loading = false;
    self.postMessage({ type: 'error', text: err.message || String(err) });
  }
}

async function generate(job) {
  const tokenLimit = job.maxNewTokens > 0 ? job.maxNewTokens : 400;
  const requestId = job.requestId;
  try {
    self.postMessage({ type: 'generating', requestId });

    // Kontext proportional auf das Budget kürzen (Top-3 aus e5-large-Retrieval).
    let ctxText = '';
    const ctx = (job.context || []).slice(0, 3);
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

    // Token-Streaming
    let streamer = null;
    if (TextStreamer && generator.tokenizer) {
      try {
        streamer = new TextStreamer(generator.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (txt) => { if (txt) self.postMessage({ type: 'token', text: txt, requestId }); }
        });
      } catch (_) { streamer = null; }
    }

    const timeoutMs = Math.max(60000, tokenLimit * 220 + 15000);
    const genOpts = {
      max_new_tokens: tokenLimit, do_sample: false,      // greedy = faktentreuer
      repetition_penalty: 1.0,
    };
    if (streamer) genOpts.streamer = streamer;
    const result = await withTimeout(generator(messages, genOpts), timeoutMs);

    // Antwort extrahieren (Chat-Template -> letzter assistant-Turn)
    let output = result[0] && result[0].generated_text, answer = '';
    if (Array.isArray(output)) {
      const m = output.filter(x => x.role === 'assistant').pop();
      answer = m ? (m.content || '') : (output[output.length - 1]?.content || '');
    } else { answer = String(output || ''); }

    // Bereinigung (Think-/Special-Tokens, Markdown-Reste)
    answer = answer.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    answer = answer.split('<|im_end|>')[0].split('</s>')[0].split('<end_of_turn>')[0].trim();
    answer = answer.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/^#{1,4}\s+/gm, '').trim();

    const noInfo = /\s*[-–•]?\s*(?:dazu\s+)?enthalten\s+die\s+dokumente\s+keine\s+(?:konkrete\s+)?angabe[.…]*\s*$/i;
    if (noInfo.test(answer)) {
      const stripped = answer.replace(noInfo, '').trim();
      if (stripped.length >= 80) answer = stripped;
    }

    self.postMessage({ type: 'done', text: answer || '(leere Antwort)', requestId });
  } catch (err) {
    self.postMessage({ type: 'error', text: err.message || String(err), requestId });
  }
}

function withTimeout(p, ms) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('Timeout nach ' + (ms / 1000) + 's')), ms))]);
}
function post(type, text, progress) { self.postMessage({ type, text, progress }); }
