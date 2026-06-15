# -*- coding: utf-8 -*-
"""
armachat Eval-Harness (pure Python + pypdf, AppLocker-tauglich).
Misst Retrieval-Recall@k VOR vs. NACH der Chunking/BM25-Optimierung.
Aufruf:  python eval_harness.py  (PDFs in ./data/ oder Pfad unten anpassen)

WICHTIG: Dies misst nur RETRIEVAL (findet die Suche den richtigen Chunk?),
nicht die LLM-Antwortqualitaet. Erweitert das golden-Set mit EUREN echten Fragen.
"""
import re, math, glob
from pypdf import PdfReader

PDF_GLOB = "./data/*.pdf"   # <-- anpassen

# ---------- Golden-Set: (Frage, erwarteter Antwort-Token) ----------
# Ergaenzt das mit euren realen Fragen + dem Wort/der Zahl, die in der Antwort stehen MUSS.
GOLDEN = [
 ("Wie oft fuehrt die MAA eine Betriebspruefung durch?", "24 Monate"),
 ("Welches KESO-Produkt hat Patentschutz bis 2044?", "8000-Omega-3"),
 ("Was kostet ein ZUKO XXI Zutrittspunkt?", "20'000"),
 ("Welche Risikostufe gilt fuer Risikoindex 9 bis 14?", "Low"),
 ("Innerhalb welcher Frist sind Maengel der MAA zu melden?", "72 Stunden"),
 ("Welche ISO-Norm legitimiert das Ideenmanagement?", "9001"),
 ("Welche Widerstandsklasse muss ein Schluesselrohr mindestens haben?", "RC 3"),
 ("Was bedeutet die Abkuerzung VS?", "Verantwortliche Stelle"),
]

def extract(src):
    r=PdfReader(src); p=[]
    for pg in r.pages:
        try:t=pg.extract_text() or ""
        except Exception:t=""
        if t.strip():p.append(t)
    return "\n".join(p)
def tok(s): return re.findall(r'[\wäöüÄÖÜ./-]+', s.lower())

# ===== ALT: euer urspruengliches Chunking =====
def chunks_old(text, size=350, ov=80):
    t=text.replace("\xad",""); t=re.sub(r"-\s*\n\s*","",t); t=re.sub(r"\s+"," ",t).strip()
    step=max(1,size-ov); out=[]; i=0
    while i<len(t):
        pc=t[i:i+size].strip()
        if len(pc)>=20: out.append(pc)
        if i+size>=len(t): break
        i+=step
    return out

# ===== NEU: strukturbewusst =====
_TOC=re.compile(r'\.{5,}\s*\d+\s*$'); _HEAD=re.compile(r'^\s*(\d+(?:\.\d+){0,3})\s+([A-ZÄÖÜ].{2,80})$')
def chunks_new(text, src, target=1100, hard=1600):
    text=text.replace("\xad",""); text=re.sub(r"-\s*\n\s*","",text); lines=[]
    for ln in text.split("\n"):
        ln=ln.strip()
        if not ln or _TOC.search(ln) or re.match(r'^(MS ID/Ver|Dok-ID/Vers)\s',ln) or re.match(r'^\d{1,3}$',ln): continue
        lines.append(ln)
    ch=[]; cur=[]; head=""; L=0
    def flush():
        nonlocal cur,L
        if cur:
            b=" ".join(cur).strip()
            if len(b)>=40: ch.append((f"[{src} | {head}] " if head else f"[{src}] ")+b)
        cur=[]; L=0
    for ln in lines:
        m=_HEAD.match(ln)
        if m: flush(); head=f"{m.group(1)} {m.group(2)}"; cur=[ln]; L=len(ln); continue
        if L+len(ln)>hard: flush()
        cur.append(ln); L+=len(ln)+1
        if L>=target and ln.endswith(('.',';',':')): flush()
    flush(); return ch

class PureBM25:
    def __init__(self, corp, k1=1.5, b=0.75):
        self.k1,self.b=k1,b; self.N=len(corp); self.dl=[len(d) for d in corp]
        self.avgdl=sum(self.dl)/self.N if self.N else 0; self.tf=[]; df={}
        for d in corp:
            f={}
            for w in d: f[w]=f.get(w,0)+1
            self.tf.append(f)
            for w in f: df[w]=df.get(w,0)+1
        self.idf={w:math.log(1+(self.N-n+0.5)/(n+0.5)) for w,n in df.items()}
    def topk(self,q,k=5):
        qt=tok(q); sc=[0.0]*self.N
        for i in range(self.N):
            f=self.tf[i]; dl=self.dl[i]; s=0.0
            for w in qt:
                tf=f.get(w)
                if not tf: continue
                s+=self.idf.get(w,0)*(tf*(self.k1+1))/(tf+self.k1*(1-self.b+self.b*dl/(self.avgdl or 1)))
            sc[i]=s
        return sorted(range(self.N),key=lambda i:sc[i],reverse=True)[:k]

def run(label, corpus):
    bm=PureBM25([tok(c) for c in corpus]); hit=0
    for q,g in GOLDEN:
        ok=any(g.lower() in corpus[i].lower() for i in bm.topk(q,5)); hit+=ok
    junk=sum(1 for c in corpus if c.count('.')>30 and len(re.sub(r'[.\s\d]','',c))<40)
    print(f"{label:28} Chunks={len(corpus):4}  ToC-Muell={junk:3}  BM25-Recall@5={hit}/{len(GOLDEN)}")

old=[]; new=[]
for f in glob.glob(PDF_GLOB):
    t=extract(f); src=f.split('/')[-1].split('\\')[-1].split('_')[0]
    old+=chunks_old(t); new+=chunks_new(t,src)
if not old:
    print(f"Keine PDFs in {PDF_GLOB} gefunden. Pfad anpassen.")
else:
    print("="*70); run("ALT (350-Zeichen, flach)", old); run("NEU (strukturbewusst)", new); print("="*70)
