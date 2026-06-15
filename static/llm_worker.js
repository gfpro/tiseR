/**
 * armachat — LLM-Worker  (gemma-4-E4B-it, q4f16 WebGPU / uint8 WASM-Fallback)
 * Abgeleitet aus LoKI loki_worker.js (bewährte Lade-Sequenz + Output-Parsing).
 * Vereinfacht: EIN strikter RAG-Prompt, KEINE Fragetyp-Klassifikation,
 * KEINE deterministische Faktenschicht. Greedy decoding für Faktentreue.
 *
 * Ablegen unter:  C:\armachat\static\llm_worker.js
 */
let MODEL_NAME  = 'gemma-4-E4B-it';
let MODEL_DTYPE = 'q4f16';     // WebGPU
let WASM_DTYPE  = 'uint8';     // CPU-Fallback
const ORIGIN    = self.location.origin;
const TJS_URL   = ORIGIN + '/static/transformersjs-420/transformers.min.js';
const WASM_BASE = ORIGIN + '/static/transformersjs-420/';
let MODEL_BASE  = ORIGIN + '/models/' + MODEL_NAME;
let PROMPT_BUDGET = 3000;      // Gesamt-Zeichenbudget für den Kontext (E4B)

let generator = null, loading = false, generating = false;
const queue = [];

const SYSTEM_PROMPT =
  'Du bist ein Assistent für interne Prozessdokumentation. Antworte auf Deutsch, ' +
  'nur auf Basis der bereitgestellten Dokumente. Du darfst Informationen aus ' +
  'mehreren Stellen kombinieren und zusammenfassen. Enthalten die Dokumente keine ' +
  'Grundlage für die Antwort, sage das offen. Erfinde keine Fakten. ' +
  'Keine Emojis, keine Markdown-Formatierung.';

// fetch()-Override (siehe embed_worker.js)
const _origFetch = self.fetch.bind(self);
self.fetch = function (url, opts) {
  let u = (typeof url === 'string') ? url : (url && url.url) ? url.url : String(url);
  if (u.indexOf(ORIGIN) >= 0 || u.endsWith('.wasm') || u.endsWith('.mjs')) return _origFetch(url, opts);
  const fname = u.split('?')[0].split('/').pop();
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
    if (MODEL_NAME === 'gemma-4-E4B-it') PROMPT_BUDGET = 3000;
    else if (MODEL_NAME === 'gemma-4-E2B-it') PROMPT_BUDGET = 2400;
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

async function loadModel() {
  if (generator || loading) return;
  loading = true;
  post('status', 'Transformers.js lädt…', 0);
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
      env.backends.onnx.wasm.numThreads = 1;
      env.backends.onnx.wasm.proxy = false;
    }

    // WebGPU versuchen, sonst CPU/uint8
    let device = 'wasm', dtype = WASM_DTYPE;
    if (self.navigator && self.navigator.gpu) {
      try { if (await self.navigator.gpu.requestAdapter()) { device = 'webgpu'; dtype = MODEL_DTYPE; } }
      catch (_) {}
    }
    post('status', 'LLM lädt (' + device.toUpperCase() + ')…', 5);

    try {
      generator = await pipeline('text-generation', MODEL_NAME, {
        dtype, device,
        progress_callback: (p) => post('status', 'LLM lädt ' + (p.progress ? Math.round(p.progress) : 0) + '%…', p.progress || 0)
      });
    } catch (err) {
      if (device === 'webgpu') {
        post('status', 'GPU n/a — lade auf CPU (uint8)…', 5);
        generator = await pipeline('text-generation', MODEL_NAME, {
          dtype: WASM_DTYPE, device: 'wasm',
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

    // Kontext proportional auf das Budget kürzen (LoKI A2)
    let ctxText = '';
    const ctx = (job.context || []).slice(0, 5);
    if (ctx.length) {
      const totalRaw = ctx.reduce((s, c) => s + (c.content || '').length, 0);
      const perChunk = totalRaw <= PROMPT_BUDGET ? totalRaw : Math.max(200, Math.floor(PROMPT_BUDGET / ctx.length));
      ctxText = ctx.map(c => '[' + (c.source || 'DOK') + ']\n' + (c.content || '').slice(0, perChunk)).join('\n\n---\n\n');
    }

    // systemPrompt aus dem Job verwenden (ermoeglicht Chat- vs. RAG-Modus von index.html aus)
    const sysPrompt = job.systemPrompt || SYSTEM_PROMPT;

    // Kurzzeitgedaechtnis: vorherige Q&A-Turns einfuegen.
    // Im RAG-Modus konkurriert die Historie mit dem Dokumentbudget -> bewusst kuerzer.
    const rawHist = Array.isArray(job.history) ? job.history : [];
    const maxTurns = ctxText ? 2 : 4;                 // RAG: 1 Paar, Chat: 2 Paare
    const perHistChars = ctxText ? 220 : 400;         // einzelne Historien-Nachricht kappen
    const histMsgs = rawHist.slice(-maxTurns).map(h => ({
      role: (h.role === 'assistant' ? 'assistant' : 'user'),
      content: String(h.content || '').slice(0, perHistChars)
    }));

    const currentUser = ctxText
      ? { role: 'user', content: 'Dokumente:\n\n' + ctxText + '\n\n---\nFrage: ' + job.prompt }
      : { role: 'user', content: job.prompt };
    const messages = [{ role: 'system', content: sysPrompt }, ...histMsgs, currentUser];

    const timeoutMs = Math.max(90000, tokenLimit * 650 + 20000);
    const result = await withTimeout(generator(messages, {
      max_new_tokens: tokenLimit, do_sample: false,      // greedy = faktentreuer
      repetition_penalty: 1.0,                            // 1.0 = keine Ziffern werden verschluckt (9001 bleibt 9001)
    }), timeoutMs);

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

    // LFM2 haengt manchmal reflexhaft eine Fehlanzeige-Floskel an eine bereits
    // vollstaendige Antwort. Diese NUR kappen, wenn klar substantieller Text
    // davor steht — steht sie allein (echte Fehlanzeige), bleibt sie unangetastet.
    const noInfo = /\s*[-–•]?\s*(?:dazu\s+)?enthalten\s+die\s+dokumente\s+keine\s+(?:konkrete\s+)?angabe[.…]*\s*$/i;
    if (noInfo.test(answer)) {
      const stripped = answer.replace(noInfo, '').trim();
      if (stripped.length >= 80) answer = stripped;   // genug Substanz davor -> Floskel war redundant
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
