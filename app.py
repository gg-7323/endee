"""
app.py — Notes Q&A Bot (Full Version)
Features:
  1. RAG Q&A          — Ask questions, get answers from your notes
  2. Voice Mode       — Speak your question, hear the answer
  3. Concept Linking  — Discover relationships between topics
  4. Flashcards       — Auto-generated study cards from notes
  5. Mind Map         — Visual topic map of your notes
  6. Story Mode       — Complex concepts as cartoon character stories
  7. Quiz Generator   — MCQ quiz from notes
  8. Notes Summarizer — Structured summary of key concepts
"""

import os
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Notes Q&A Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif}
.stApp{background:linear-gradient(135deg,#0f1117 0%,#1a1d2e 100%)}
#MainMenu,footer,header{visibility:hidden}
.app-header{text-align:center;padding:1.5rem 0 1rem;border-bottom:1px solid #2a2d3e;margin-bottom:1.5rem}
.app-title{font-size:2.2rem;font-weight:700;background:linear-gradient(90deg,#7c6fff,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.app-subtitle{color:#6b7280;font-size:.9rem;font-weight:300;margin-top:.3rem}
.chat-user{background:linear-gradient(135deg,#1e2035,#252840);border:1px solid #3730a3;border-radius:16px 16px 4px 16px;padding:1rem 1.2rem;margin:.6rem 0;color:#e2e8f0;font-size:.93rem}
.chat-bot{background:linear-gradient(135deg,#111827,#1f2937);border:1px solid #374151;border-radius:4px 16px 16px 16px;padding:1rem 1.2rem;margin:.6rem 0;color:#e2e8f0;font-size:.93rem;line-height:1.7}
.source-chip{display:inline-flex;align-items:center;gap:5px;background:#111827;border:1px solid #1f2937;border-left:3px solid #7c6fff;border-radius:7px;padding:5px 10px;font-size:.78rem;color:#9ca3af;font-family:'JetBrains Mono',monospace;margin:3px}
.source-score{color:#7c6fff;font-weight:600}
.feature-card{background:#111827;border:1px solid #1f2937;border-radius:14px;padding:1.3rem;margin-bottom:1rem}
.feature-card h3{color:#e2e8f0;font-size:.95rem;margin-bottom:.4rem}
.feature-card p{color:#6b7280;font-size:.82rem;line-height:1.5;margin-bottom:1rem}
.fc-container{perspective:1000px;cursor:pointer;width:100%;height:160px;margin:8px 0}
.fc-inner{position:relative;width:100%;height:100%;transition:transform .5s;transform-style:preserve-3d}
.fc-inner.flipped{transform:rotateY(180deg)}
.fc-face{position:absolute;width:100%;height:100%;backface-visibility:hidden;border-radius:12px;display:flex;align-items:center;justify-content:center;padding:1rem;text-align:center;flex-direction:column;gap:.4rem}
.fc-front{background:linear-gradient(135deg,#1e1b4b,#312e81);border:1px solid #4338ca;color:#c7d2fe;font-size:.9rem;font-weight:500}
.fc-back{background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid #047857;color:#a7f3d0;font-size:.85rem;transform:rotateY(180deg)}
.fc-hint{font-size:.7rem;color:#818cf8}
.fc-diff{padding:2px 8px;border-radius:10px;font-size:.68rem;font-weight:600;align-self:flex-end}
.diff-easy{background:#064e3b;color:#34d399}
.diff-medium{background:#1e3a5f;color:#60a5fa}
.diff-hard{background:#450a0a;color:#f87171}
.rel-box{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem}
.concept-node{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:20px;font-size:.88rem;font-weight:600}
.node-a{background:#1e1b4b;border:2px solid #6d28d9;color:#c4b5fd}
.node-b{background:#1e3a5f;border:2px solid #1d4ed8;color:#93c5fd}
.rel-label{background:#1f2d1f;border:1px solid #166534;color:#4ade80;padding:4px 12px;border-radius:20px;font-size:.78rem;font-weight:600}
.strength-bar{height:5px;border-radius:3px;background:linear-gradient(90deg,#7c6fff,#4ade80)}
.scene-card{background:#111827;border:1px solid #2a2d3e;border-radius:12px;padding:1.2rem;margin:8px 0;position:relative}
.scene-num{position:absolute;top:10px;right:12px;width:26px;height:26px;border-radius:50%;background:#7c6fff;color:#fff;font-size:.75rem;font-weight:700;display:flex;align-items:center;justify-content:center}
.dialogue-bubble{background:#1f2937;border-radius:10px;padding:8px 12px;margin:6px 0;font-size:.83rem;color:#d1d5db}
.char-name{color:#7c6fff;font-weight:600;font-size:.78rem;margin-bottom:3px}
.mm-node{border-radius:8px;padding:4px 12px;font-size:.78rem;font-weight:600;display:inline-block;margin:2px}
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600}
.badge-ok{background:#064e3b;color:#34d399}
.badge-err{background:#450a0a;color:#f87171}
[data-testid="stSidebar"]{background:#0d0f1a;border-right:1px solid #1f2937}
.stButton>button{background:linear-gradient(135deg,#7c6fff,#6d28d9)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Sora',sans-serif!important;font-weight:600!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 4px 15px rgba(124,111,255,.4)!important}
</style>
""", unsafe_allow_html=True)

from src.endee_client import ping
from src.ingest import ingest_file
from src.rag import answer_question, generate_quiz, summarize_notes

def init():
    defaults = {
        "chat_history":[],"selected_index":None,"ingested_files":{},
        "show_sources":True,"top_k":5,"fc_index":0,"fc_flipped":False,
        "flashcards":[],"mindmap":None,"story":None,
        "concept_link_result":None,"concepts_list":[],
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
init()

st.markdown("""<div class="app-header">
  <p class="app-title">📚 Notes Q&A Bot</p>
  <p class="app-subtitle">RAG · Voice Mode · Concept Linking · Flashcards · Mind Map · Story Mode</p>
</div>""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    endee_ok = ping()
    api_ok = bool(os.getenv("GROQ_API_KEY","").strip()) and os.getenv("GROQ_API_KEY","") != "your_groq_api_key_here"
    st.markdown(f'<span class="badge {"badge-ok" if endee_ok else "badge-err"}">{"● Endee online" if endee_ok else "● Endee offline"}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge {"badge-ok" if api_ok else "badge-err"}">{"● Groq API ready" if api_ok else "● Groq key missing"}</span>', unsafe_allow_html=True)
    if not endee_ok:
        st.code("docker compose up -d", language="bash")
    st.divider()
    st.markdown("### 📂 Upload Notes")
    uploaded = st.file_uploader("PDF, TXT or MD", type=["pdf","txt","md"])
    if uploaded:
        overwrite = st.checkbox("Re-ingest if exists", False)
        if st.button("🚀 Ingest", use_container_width=True):
            if not endee_ok: st.error("Endee is offline.")
            else:
                pb=st.progress(0); st_txt=st.empty()
                def cb(c,t,m): pb.progress(c/t); st_txt.markdown(f"<small style='color:#9ca3af'>{m}</small>",unsafe_allow_html=True)
                try:
                    res=ingest_file(uploaded.read(),uploaded.name,overwrite=overwrite,progress_callback=cb)
                    st.session_state.ingested_files[uploaded.name]=res["index_name"]
                    st.session_state.selected_index=res["index_name"]
                    st.session_state.chat_history=[]; st.session_state.flashcards=[]; st.session_state.mindmap=None; st.session_state.story=None
                    pb.progress(1.0); st_txt.empty(); st.success(f"✅ {res['chunk_count']} chunks in Endee!"); st.rerun()
                except Exception as e: pb.empty(); st_txt.empty(); st.error(str(e))
    st.divider()
    if st.session_state.ingested_files:
        st.markdown("### 📋 Active Notes")
        sel=st.selectbox("Notes file:",list(st.session_state.ingested_files.keys()))
        new_idx=st.session_state.ingested_files[sel]
        if new_idx!=st.session_state.selected_index:
            st.session_state.selected_index=new_idx; st.session_state.chat_history=[]
            st.session_state.flashcards=[]; st.session_state.mindmap=None; st.session_state.story=None
        st.markdown(f"<small style='color:#7c6fff;font-family:monospace'>{st.session_state.selected_index}</small>",unsafe_allow_html=True)
        st.divider()
    st.markdown("### 🎛️ Settings")
    st.session_state.top_k=st.slider("Retrieve top_k chunks",2,10,5)
    st.session_state.show_sources=st.toggle("Show source chunks",True)
    st.divider()
    st.markdown("### 🧪 Sample Notes")
    samples=list(Path("data/sample_notes").glob("*.txt"))
    if samples and endee_ok:
        s=st.selectbox("Load sample:",[f.name for f in samples],key="samp")
        if st.button("📥 Load Sample",use_container_width=True):
            fb=(Path("data/sample_notes")/s).read_bytes()
            with st.spinner("Ingesting..."):
                try:
                    res=ingest_file(fb,s,overwrite=True)
                    st.session_state.ingested_files[s]=res["index_name"]
                    st.session_state.selected_index=res["index_name"]
                    st.session_state.chat_history=[]; st.session_state.flashcards=[]; st.session_state.mindmap=None; st.session_state.story=None
                    st.success(f"✅ {res['chunk_count']} chunks!"); st.rerun()
                except Exception as e: st.error(str(e))

# ─── No file ───────────────────────────────────────────────────────────────────
if not st.session_state.selected_index:
    st.markdown("""<div style='text-align:center;padding:3rem 2rem;color:#4b5563'>
      <div style='font-size:4rem'>📖</div>
      <h3 style='color:#6b7280;font-weight:400;margin:1rem 0'>Upload notes to get started</h3>
    </div>""", unsafe_allow_html=True)
    cols=st.columns(4)
    for col,icon,title,desc in [
        (cols[0],"💬","RAG Q&A","Answers grounded in your notes"),
        (cols[1],"🎤","Voice Mode","Speak + hear answers"),
        (cols[2],"🔗","Concept Linking","See how topics relate"),
        (cols[3],"🃏","Flashcards+Map","Cards, mindmaps, stories"),
    ]:
        col.markdown(f"""<div style='background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem;text-align:center'>
        <div style='font-size:1.8rem'>{icon}</div><h4 style='color:#e2e8f0;margin:.4rem 0;font-size:.9rem'>{title}</h4>
        <p style='color:#6b7280;font-size:.78rem'>{desc}</p></div>""",unsafe_allow_html=True)
    st.stop()

idx=st.session_state.selected_index
tabs=st.tabs(["💬 Q&A","🎤 Voice","🔗 Concept Links","🃏 Flashcards","🗺️ Mind Map","📖 Story Mode","🛠️ Tools"])

# ── TAB 1: Q&A ─────────────────────────────────────────────────────────────────
with tabs[0]:
    if not st.session_state.chat_history:
        st.markdown("<p style='color:#4b5563;font-size:.83rem;margin-bottom:.5rem'>💡 Suggested questions:</p>",unsafe_allow_html=True)
        scols=st.columns(2)
        for i,s in enumerate(["What are the key concepts?","Explain the most important formula.","What definitions should I memorize?","Give me a full summary."]):
            if scols[i%2].button(s,key=f"s{i}",use_container_width=True): st.session_state._pq=s
    for msg in st.session_state.chat_history:
        if msg["role"]=="user":
            st.markdown(f'<div class="chat-user">🧑 <strong>You:</strong> {msg["content"]}</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>',unsafe_allow_html=True)
            if st.session_state.show_sources and msg.get("chunks"):
                with st.expander(f"📎 {len(msg['chunks'])} source chunks from Endee"):
                    for c in msg["chunks"]:
                        st.markdown(f'<div class="source-chip"><span class="source-score">#{c["chunk_index"]}·{c["score"]:.0%}</span> · {c["source_file"]} · {c["text"][:120]}...</div>',unsafe_allow_html=True)
    def do_ask(q):
        hist=[{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_history[-6:]]
        with st.spinner("🔍 Retrieving from Endee · Generating..."):
            res=answer_question(idx,q,top_k=st.session_state.top_k,chat_history=hist or None)
        st.session_state.chat_history.append({"role":"user","content":q})
        st.session_state.chat_history.append({"role":"assistant","content":res["answer"],"chunks":res["retrieved_chunks"]})
        st.rerun()
    if hasattr(st.session_state,"_pq"):
        pq=st.session_state._pq; del st.session_state._pq; do_ask(pq)
    if q:=st.chat_input("Ask anything about your notes..."): do_ask(q)
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"): st.session_state.chat_history=[]; st.rerun()

# ── TAB 2: VOICE ───────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🎤 Voice Mode")
    st.markdown("<p style='color:#6b7280;font-size:.88rem'>Speak your question → RAG pipeline → Claude answers → gTTS reads it aloud.</p>",unsafe_allow_html=True)
    col_l,col_r=st.columns(2)
    with col_l:
        st.markdown("""<div class="feature-card"><h3>🎙️ Step 1 — Speak</h3>
        <p>Uses your browser's Web Speech API (Chrome/Edge). Click and speak clearly.</p></div>""",unsafe_allow_html=True)
        st.components.v1.html("""
        <style>
          body{margin:0;font-family:Sora,sans-serif}
          #vbtn{padding:12px 24px;background:linear-gradient(135deg,#7c6fff,#6d28d9);color:#fff;border:none;
                border-radius:24px;cursor:pointer;font-size:14px;font-weight:700;box-shadow:0 4px 16px rgba(124,111,255,.4)}
          #vbtn:disabled{opacity:.5;cursor:not-allowed}
          #vstatus{margin-top:10px;font-size:12px;color:#9ca3af;min-height:18px}
          #vtrans{margin-top:8px;padding:10px;background:#1f2937;border:1px solid #374151;border-radius:8px;
                  color:#e2e8f0;font-size:13px;min-height:36px;display:none;word-break:break-word}
          #vcopy{margin-top:6px;padding:5px 14px;background:#111827;color:#7c6fff;border:1px solid #7c6fff;
                 border-radius:7px;cursor:pointer;font-size:11px;display:none}
        </style>
        <button id="vbtn" onclick="go()">🎤 Tap to Speak</button>
        <div id="vstatus">Click above and speak your question</div>
        <div id="vtrans"></div>
        <button id="vcopy" onclick="copyIt()">📋 Copy transcript</button>
        <script>
        let tr='';
        function go(){
          if(!('webkitSpeechRecognition' in window||'SpeechRecognition' in window)){
            document.getElementById('vstatus').textContent='⚠️ Use Chrome or Edge';return;
          }
          const SR=window.SpeechRecognition||window.webkitSpeechRecognition,r=new SR();
          r.lang='en-US';r.interimResults=true;
          const b=document.getElementById('vbtn'),s=document.getElementById('vstatus'),t=document.getElementById('vtrans'),c=document.getElementById('vcopy');
          r.onstart=()=>{b.textContent='🔴 Listening...';b.disabled=true;s.textContent='Speak now...';};
          r.onresult=(e)=>{let fi='',in_='';for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)fi+=e.results[i][0].transcript;else in_+=e.results[i][0].transcript;}tr=fi||in_;t.textContent=tr;t.style.display='block';};
          r.onend=()=>{b.textContent='🎤 Tap to Speak';b.disabled=false;if(tr){s.textContent='✅ Done! Copy below.';c.style.display='inline-block';}else s.textContent='No speech detected.';};
          r.onerror=(e)=>{b.textContent='🎤 Tap to Speak';b.disabled=false;s.textContent='❌ '+e.error;};
          r.start();
        }
        function copyIt(){navigator.clipboard.writeText(tr).then(()=>document.getElementById('vstatus').textContent='✅ Copied! Paste in Q&A tab.');}
        </script>""",height=210)
        st.markdown("**Or type for voice response:**")
        vq=st.text_input("Question:",placeholder="e.g. What is Newton's Second Law?",key="vq")
        ask_v=st.button("🔍 Answer + Speak",use_container_width=True)
    with col_r:
        st.markdown("""<div class="feature-card"><h3>🔊 Step 2 — Listen</h3>
        <p>Endee retrieves relevant chunks → Claude generates answer → gTTS converts to speech → plays in browser.</p></div>""",unsafe_allow_html=True)
        if ask_v and vq.strip():
            with st.spinner("🔍 RAG pipeline running..."):
                res=answer_question(idx,vq,top_k=st.session_state.top_k)
            st.markdown(f'<div class="chat-bot">🤖 <strong>Answer:</strong><br>{res["answer"]}</div>',unsafe_allow_html=True)
            with st.spinner("🔊 Generating audio..."):
                try:
                    from src.voice import text_to_speech,make_audio_html
                    audio=text_to_speech(res["answer"][:1500])
                    st.markdown("**🔊 Audio answer:**")
                    st.markdown(make_audio_html(audio,autoplay=True),unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"TTS unavailable ({e}). Showing text only.")
            if st.session_state.show_sources and res["retrieved_chunks"]:
                with st.expander(f"📎 {len(res['retrieved_chunks'])} Endee chunks"):
                    for c in res["retrieved_chunks"]:
                        st.markdown(f'<div class="source-chip"><span class="source-score">{c["score"]:.0%}</span> Chunk {c["chunk_index"]} · {c["text"][:100]}...</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div style='background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem'>
    <p style='color:#6b7280;font-size:.82rem;margin:0'><strong style='color:#9ca3af'>Voice pipeline:</strong><br>
    1. 🎤 Browser Web Speech API → transcript<br>2. 🔍 Embed query → Endee ANN search → top-K chunks<br>
    3. 🤖 Claude RAG → grounded answer<br>4. 🔊 gTTS → MP3 → browser playback</p></div>""",unsafe_allow_html=True)

# ── TAB 3: CONCEPT LINKING ─────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🔗 Concept Linking")
    st.markdown("<p style='color:#6b7280;font-size:.88rem'>Discover how topics in your notes relate. Multi-concept retrieval + reasoning layer — not just search.</p>",unsafe_allow_html=True)
    c1,c2=st.columns([3,1])
    with c1:
        lq=st.text_input("Relationship question:",placeholder='"How is Newton\'s 2nd Law related to Momentum?" or "What connects friction and inertia?"',key="lq")
    with c2:
        st.markdown("<br>",unsafe_allow_html=True)
        ask_l=st.button("🔗 Find Connection",use_container_width=True)
    ec1,ec2,ec3=st.columns(3)
    if ec1.button("Newton's 2nd Law ↔ Momentum",key="ex1",use_container_width=True): st.session_state._lq="How is Newton's Second Law related to Momentum?"
    if ec2.button("Static ↔ Kinetic Friction",key="ex2",use_container_width=True): st.session_state._lq="What is the relationship between static friction and kinetic friction?"
    if ec3.button("Inertia ↔ Force",key="ex3",use_container_width=True): st.session_state._lq="How does inertia relate to force?"
    q2use=lq if ask_l and lq.strip() else st.session_state.pop("_lq",None)
    if q2use:
        with st.spinner("🧠 Dual-concept retrieval from Endee · Reasoning..."):
            from src.concept_linker import answer_concept_link_question
            st.session_state.concept_link_result=answer_concept_link_question(idx,q2use)
    if st.session_state.concept_link_result:
        r=st.session_state.concept_link_result; rel=r["relationship"]; ca,cb=r["concepts"][0],r["concepts"][1]
        st.markdown("---")
        st.markdown(f"""<div class="rel-box">
          <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap'>
            <span class="concept-node node-a">🔵 {ca}</span>
            <div style='display:flex;align-items:center;gap:8px;color:#6b7280;font-size:.82rem'>
              ──── <span class="rel-label">{rel.get("relationship","related_to").replace("_"," ")}</span> ────▶
            </div>
            <span class="concept-node node-b">🔷 {cb}</span>
          </div>
          <div style='margin-top:12px'>
            <div style='display:flex;justify-content:space-between;font-size:.78rem;color:#6b7280;margin-bottom:4px'>
              <span>Connection strength</span><span style='color:#7c6fff;font-weight:600'>{rel.get("strength",0.5):.0%}</span>
            </div>
            <div style='background:#1f2937;border-radius:4px;height:5px'>
              <div class="strength-bar" style='width:{rel.get("strength",0.5)*100:.0f}%'></div></div>
            <p style='color:#9ca3af;font-size:.83rem;margin-top:10px'><strong style='color:#7c6fff'>Key link:</strong> {rel.get("key_link","–")}</p>
          </div></div>""",unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot">🧠 <strong>Explanation:</strong><br>{r["answer"]}</div>',unsafe_allow_html=True)
        if st.session_state.show_sources and r.get("chunks_used"):
            with st.expander(f"📎 {len(r['chunks_used'])} chunks (both concepts)"):
                for c in r["chunks_used"]:
                    st.markdown(f'<div class="source-chip"><span class="source-score">{c["score"]:.0%}</span> {c["source_file"]} · {c["text"][:110]}...</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 🗂️ Extract All Concepts")
    if st.button("🧩 Extract Concept List"):
        with st.spinner("Sampling notes from Endee..."):
            from src.concept_linker import extract_concepts
            st.session_state.concepts_list=extract_concepts(idx)
    if st.session_state.concepts_list:
        st.markdown("**Concepts found in your notes:**")
        ccols=st.columns(4)
        for i,c in enumerate(st.session_state.concepts_list):
            ccols[i%4].markdown(f'<span class="mm-node" style="background:#1e1b4b;color:#c4b5fd;border:1px solid #4338ca">{c}</span>',unsafe_allow_html=True)

# ── TAB 4: FLASHCARDS ──────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🃏 Flashcard Generator")
    st.markdown("<p style='color:#6b7280;font-size:.88rem'>Auto-generate study flashcards from your notes. Click any card to flip and reveal the answer.</p>",unsafe_allow_html=True)
    ca,cb=st.columns([1,2])
    with ca:
        n_cards=st.slider("Number of cards",4,20,8)
        difficulty=st.selectbox("Difficulty",["mixed","easy","medium","hard"])
        if st.button("⚡ Generate Flashcards",use_container_width=True):
            with st.spinner("📡 Endee retrieval · Generating cards..."):
                from src.flashcards import generate_flashcards
                st.session_state.flashcards=generate_flashcards(idx,n_cards,difficulty)
                st.session_state.fc_index=0
            st.success(f"✅ {len(st.session_state.flashcards)} cards ready!")
    with cb:
        if st.session_state.flashcards:
            cards=st.session_state.flashcards; ci=st.session_state.fc_index
            card=cards[min(ci,len(cards)-1)]
            dc={"easy":"diff-easy","medium":"diff-medium","hard":"diff-hard"}.get(card.get("difficulty","medium"),"diff-medium")
            st.markdown(f"""<div class="fc-container" onclick="document.getElementById('fc').classList.toggle('flipped')">
              <div class="fc-inner" id="fc">
                <div class="fc-face fc-front">
                  <span class="fc-diff {dc}">{card.get('difficulty','?').upper()}</span>
                  <div>{card.get('front','')}</div>
                  <span class="fc-hint">💡 {card.get('hint','Tap to flip')}</span>
                </div>
                <div class="fc-face fc-back"><div>{card.get('back','')}</div></div>
              </div></div>""",unsafe_allow_html=True)
            n1,n2,n3=st.columns([1,2,1])
            with n1:
                if st.button("◀ Prev") and ci>0: st.session_state.fc_index-=1; st.rerun()
            with n2:
                st.markdown(f"<p style='text-align:center;color:#6b7280;font-size:.85rem;margin-top:.5rem'>Card {ci+1}/{len(cards)} · {card.get('category','')}</p>",unsafe_allow_html=True)
            with n3:
                if st.button("Next ▶") and ci<len(cards)-1: st.session_state.fc_index+=1; st.rerun()
            with st.expander("📋 All cards"):
                for i,c in enumerate(cards):
                    dc2={"easy":"#34d399","medium":"#60a5fa","hard":"#f87171"}.get(c.get("difficulty",""),"#9ca3af")
                    st.markdown(f"""<div style='background:#111827;border:1px solid #1f2937;border-radius:8px;padding:.7rem 1rem;margin:4px 0'>
                    <span style='color:{dc2};font-size:.72rem;font-weight:700'>{c.get('difficulty','').upper()}</span>
                    <p style='color:#e2e8f0;font-size:.82rem;margin:3px 0'><strong>Q:</strong> {c.get('front','')}</p>
                    <p style='color:#9ca3af;font-size:.78rem;margin:0'><strong>A:</strong> {c.get('back','')[:120]}...</p></div>""",unsafe_allow_html=True)

# ── TAB 5: MIND MAP ────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🗺️ Mind Map Generator")
    st.markdown("<p style='color:#6b7280;font-size:.88rem'>Visual hierarchical map of all topics in your notes. Generated from Endee retrieval + Claude reasoning.</p>",unsafe_allow_html=True)
    if st.button("🧠 Generate Mind Map"):
        with st.spinner("📡 Sampling Endee · Building hierarchy..."):
            from src.flashcards import generate_mindmap
            st.session_state.mindmap=generate_mindmap(idx)
        st.success("✅ Mind map ready!")
    if st.session_state.mindmap:
        mm=st.session_state.mindmap; root=mm.get("root","Notes"); children=mm.get("children",[])
        nodes_js=[{"id":0,"label":root,"color":{"background":"#7c6fff","border":"#6d28d9"},"font":{"color":"#fff","size":16},"shape":"ellipse","size":35}]
        edges_js=[]; nid=1
        for branch in children:
            bid=nid; color=branch.get("color","#4ade80")
            nodes_js.append({"id":bid,"label":branch["topic"],"color":{"background":color,"border":color},"font":{"color":"#fff","size":13},"shape":"box"})
            edges_js.append({"from":0,"to":bid,"color":{"color":color}}); nid+=1
            for sub in branch.get("children",[]):
                sid=nid
                nodes_js.append({"id":sid,"label":sub["topic"],"color":{"background":"#1f2937","border":color},"font":{"color":"#e2e8f0","size":11},"shape":"box","title":sub.get("detail","")})
                edges_js.append({"from":bid,"to":sid,"color":{"color":color},"dashes":True}); nid+=1
                for leaf in sub.get("children",[]):
                    lid=nid
                    nodes_js.append({"id":lid,"label":leaf["topic"],"color":{"background":"#111827","border":"#374151"},"font":{"color":"#9ca3af","size":10},"shape":"box"})
                    edges_js.append({"from":sid,"to":lid,"color":{"color":"#374151"},"dashes":True}); nid+=1
        st.components.v1.html(f"""
        <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
        <style>body{{margin:0;background:#0f1117}}#mm{{width:100%;height:520px;background:#0f1117;border-radius:12px;border:1px solid #1f2937}}</style>
        <div id="mm"></div>
        <script>
        const nodes=new vis.DataSet({json.dumps(nodes_js)});
        const edges=new vis.DataSet({json.dumps(edges_js)});
        new vis.Network(document.getElementById('mm'),{{nodes,edges}},{{
          layout:{{hierarchical:{{enabled:true,levelSeparation:120,nodeSpacing:100,direction:'UD'}}}},
          physics:{{enabled:false}},
          edges:{{arrows:'to',smooth:{{type:'cubicBezier'}},width:1.5}},
          nodes:{{borderWidth:1.5,font:{{face:'Sora,sans-serif'}}}},
          interaction:{{hover:true,zoomView:true,dragView:true}}
        }});
        </script>""",height=540)
        with st.expander("📋 Text outline"):
            st.markdown(f"**{root}**")
            for b in children:
                st.markdown(f"&nbsp;&nbsp;├─ **{b['topic']}**")
                for s in b.get("children",[]):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;├─ {s['topic']}"+(f" — *{s['detail']}*" if s.get('detail') else ""))
                    for l in s.get("children",[]):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ `{l['topic']}`")

# ── TAB 6: STORY MODE ──────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### 📖 Story Mode")
    st.markdown("<p style='color:#6b7280;font-size:.88rem'>Understand complex concepts through illustrated cartoon stories with memorable characters.</p>",unsafe_allow_html=True)
    sl,sr=st.columns([1,2])
    with sl:
        sc_input=st.text_input("Concept to explain:",placeholder="e.g. Newton's Second Law, Friction")
        if st.button("🔮 Generate Story",use_container_width=True):
            if sc_input.strip():
                with st.spinner("✍️ Writing cartoon story from notes..."):
                    from src.storyteller import generate_story
                    st.session_state.story=generate_story(idx,sc_input)
            else: st.warning("Enter a concept first.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📚 Suggest concepts",use_container_width=True):
            with st.spinner("Scanning notes..."):
                from src.storyteller import list_storyable_concepts
                st.session_state._story_sugg=list_storyable_concepts(idx)
        if hasattr(st.session_state,"_story_sugg"):
            for sc2 in st.session_state._story_sugg[:6]:
                if st.button(f"▶ {sc2}",key=f"sc_{sc2}",use_container_width=True):
                    with st.spinner(f"Writing story about {sc2}..."):
                        from src.storyteller import generate_story
                        st.session_state.story=generate_story(idx,sc2)
    with sr:
        if st.session_state.story:
            story=st.session_state.story
            if story.get("error"):
                st.error(story["error"])
                st.stop()
            st.markdown(f"""<div style='background:linear-gradient(135deg,#1e1b4b,#1e3a5f);border:1px solid #3730a3;border-radius:14px;padding:1.2rem;margin-bottom:1rem'>
            <h2 style='color:#e2e8f0;font-size:1.1rem;margin:0 0 .3rem'>📖 {story.get("title","")}</h2>
            <p style='color:#818cf8;font-size:.83rem;margin:0'>Concept: {story.get("concept","")}</p></div>""",unsafe_allow_html=True)
            chars=story.get("characters",[])
            if chars:
                st.markdown("**🎭 Characters:**")
                ccs=st.columns(min(len(chars),4))
                for i,ch in enumerate(chars[:4]):
                    ccs[i].markdown(f"""<div style='background:#111827;border:1px solid #1f2937;border-left:3px solid {ch.get("color","#7c6fff")};border-radius:10px;padding:.7rem;text-align:center'>
                    <div style='font-size:1.5rem'>{ch.get("emoji","🎭")}</div>
                    <div style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{ch.get("name","?")}</div>
                    <div style='color:#6b7280;font-size:.72rem'>{ch.get("represents","")}</div>
                    <div style='color:#9ca3af;font-size:.7rem;font-style:italic'>{ch.get("personality","")}</div></div>""",unsafe_allow_html=True)
            st.markdown("<br>**🎬 The Story:**",unsafe_allow_html=True)
            for scene in story.get("scenes",[]):
                st.markdown(f"""<div class="scene-card">
                <span class="scene-num">{scene.get("scene_number","?")}</span>
                <h4 style='color:#a78bfa;font-size:.88rem;margin:0 0 .3rem'>{scene.get("title","")}</h4>
                <p style='color:#6b7280;font-size:.75rem;margin:0 0 .5rem'>📍 {scene.get("setting","")}</p>
                <p style='color:#d1d5db;font-size:.85rem;line-height:1.6;margin-bottom:.6rem'>{scene.get("narration","")}</p>""",unsafe_allow_html=True)
                for line in scene.get("dialogue",[]):
                    st.markdown(f"""<div class="dialogue-bubble"><div class="char-name">{line.get("character","?")}:</div>"{line.get("line","")}"</div>""",unsafe_allow_html=True)
                if scene.get("key_concept"):
                    st.markdown(f'<p style="color:#4ade80;font-size:.75rem;margin:.4rem 0 0">🔬 {scene["key_concept"]}</p>',unsafe_allow_html=True)
                st.markdown("</div>",unsafe_allow_html=True)
            st.markdown(f"""<div style='background:#1a2e1a;border:1px solid #166534;border-radius:12px;padding:1rem;margin-top:1rem'>
            <p style='color:#4ade80;font-weight:600;font-size:.88rem;margin:0 0 .3rem'>🏆 The Moral:</p>
            <p style='color:#d1fae5;font-size:.85rem;margin:0'>{story.get("moral","")}</p></div>""",unsafe_allow_html=True)
            if story.get("fun_fact"):
                st.markdown(f"""<div style='background:#1e1b4b;border:1px solid #3730a3;border-radius:10px;padding:.8rem;margin-top:.6rem'>
                <p style='color:#818cf8;font-size:.82rem;margin:0'>⚡ <strong>Fun fact:</strong> {story["fun_fact"]}</p></div>""",unsafe_allow_html=True)
            if story.get("real_world"):
                st.info(f"🌍 **Real world:** {story['real_world']}")

# ── TAB 7: TOOLS ───────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("### 🛠️ Study Tools")
    t1,t2=st.columns(2)
    with t1:
        st.markdown("""<div class="feature-card"><h3>📝 Quiz Generator</h3>
        <p>Auto-generate MCQ quiz grounded in your uploaded notes.</p></div>""",unsafe_allow_html=True)
        nq=st.slider("Questions",3,10,5,key="nq")
        if st.button("⚡ Generate Quiz",use_container_width=True,key="gq"):
            with st.spinner("Generating quiz..."): quiz=generate_quiz(idx,num_questions=nq)
            st.markdown(quiz)
    with t2:
        st.markdown("""<div class="feature-card"><h3>📋 Notes Summarizer</h3>
        <p>Structured summary — key concepts, definitions, formulas, quick recap.</p></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("⚡ Summarize Notes",use_container_width=True,key="gs"):
            with st.spinner("Summarizing..."): summary=summarize_notes(idx)
            st.markdown(summary)
