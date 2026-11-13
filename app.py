# AI Music Composer - web app. The neural network writes the notes, you listen, watch and download.
# Start with:  python -m streamlit run app.py
#
# Notes about how it works:
# * Every time you press Compose, the LSTM writes NEW notes (token by token, with a little randomness).
#   Nothing is stored in advance. The synthesizer only turns those notes into sound.
# * You can give the model your own start (typed notes or a MIDI file) and it continues your melody.
# * Settings are inside a form and the sound is embedded in the page, so the player never freezes when
#   Streamlit re-runs the page.
import base64
import json
import os
import tempfile
import time

import streamlit as st

try:  # Windows only: lets Python play the WAV through the PC speakers, without the browser
    import winsound
except ImportError:
    winsound = None

try:  # older Streamlit versions only have components.html, newer ones use st.iframe
    import streamlit.components.v1 as components
except Exception:  # pragma: no cover
    components = None

from evaluate import analyse
from music_utils import INSTRUMENTS, tokens_to_midi, tokens_to_notes, tokens_to_wav, transpose_tokens

MODEL_PATH, DATA_DIR, OUT_DIR = "model/music_lstm.pt", "processed", "output"
MOODS = {"Calm": (0.7, 5, 0.85), "Balanced": (0.9, 8, 0.95), "Creative": (1.15, 14, 1.0)}
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAX_BATCHES = 4  # how many earlier results stay on the page

st.set_page_config(page_title="AI Music Composer", page_icon="🎼", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
footer{visibility:hidden}
.hero{padding:28px 34px;border-radius:22px;margin-bottom:18px;
      background:linear-gradient(135deg,#4c1d95 0%,#1e3a8a 55%,#0e7490 100%)}
.hero h1{margin:0;font-size:2.4rem;color:#fff;letter-spacing:-0.5px}
.hero p{margin:8px 0 0;color:#dbeafe;font-size:1.05rem}
.chip{display:inline-block;padding:3px 13px;border-radius:999px;background:rgba(255,255,255,.16);
      color:#fff;font-size:.8rem;margin:12px 8px 0 0}
div[data-testid="stMetric"]{background:#131a2b;border:1px solid #243049;padding:12px 16px;border-radius:14px}
div[data-testid="stFormSubmitButton"] button{width:100%;border:0;font-weight:700;padding:.65rem;color:#fff;
      background:linear-gradient(90deg,#8b5cf6,#06b6d4)}
div[data-testid="stFormSubmitButton"] button:hover{filter:brightness(1.12);color:#fff}
.card{background:#131a2b;border:1px solid #243049;border-radius:16px;padding:18px 20px;height:100%}
.card h4{margin:0 0 6px;color:#c4b5fd}
.card p{margin:0;color:#aab4cf;font-size:.95rem}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- player (audio + animated piano roll)
PLAYER = """
<style>
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,'Segoe UI',sans-serif;background:transparent;color:#e8ecf5}
.wrap{background:#0f1526;border:1px solid #243049;border-radius:16px;padding:14px}
canvas{width:100%;height:230px;display:block;border-radius:10px;background:#0b0f1a;cursor:pointer}
audio{width:100%;margin-top:10px;color-scheme:dark}
.hint{font-size:12px;color:#8a97b5;margin-top:6px}
</style>
<div class="wrap">
  <canvas id="c"></canvas>
  <audio id="a" controls preload="auto" src="data:audio/wav;base64,__AUDIO__"></audio>
  <div class="hint">Click on the picture to jump to that moment. The notes light up while they are played.</div>
</div>
<script>
const notes = __NOTES__, total = Math.max(__TOTAL__, 1);
const cv = document.getElementById('c'), ctx = cv.getContext('2d'), au = document.getElementById('a');
let lo = Infinity, hi = -Infinity;
for (const n of notes) { if (n[2] < lo) lo = n[2]; if (n[2] > hi) hi = n[2]; }
if (!isFinite(lo)) { lo = 48; hi = 72; }
lo -= 2; hi += 2;

function draw() {
  const w = cv.clientWidth, h = cv.clientHeight, t = au.currentTime || 0;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = '11px sans-serif';
  for (let s = 0; s <= total; s += 10) {
    const x = s / total * w;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillText(s + 's', x + 3, h - 4);
  }
  const nh = h / (hi - lo + 1);
  for (const [s, d, p] of notes) {
    const x = s / total * w, ww = Math.max(1.5, d / total * w - 1), y = h - (p - lo + 1) * nh;
    const on = t >= s && t < s + d, done = s + d <= t;
    const hue = 270 - (p - lo) / (hi - lo) * 140;
    ctx.fillStyle = on ? '#ffffff' : 'hsla(' + hue + ',85%,' + (done ? 62 : 52) + '%,' + (done ? 1 : 0.6) + ')';
    ctx.fillRect(x, y, ww, Math.max(1.5, nh - 1));
  }
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.fillRect(Math.min(t / total, 1) * w - 1, 0, 2, h);
}
function resize() {
  const r = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * r; cv.height = cv.clientHeight * r;
  ctx.setTransform(r, 0, 0, r, 0, 0);
  draw();
}
function loop() { draw(); if (!au.paused && !au.ended) requestAnimationFrame(loop); }
au.addEventListener('play', loop);
['pause', 'seeked', 'timeupdate', 'ended', 'loadedmetadata'].forEach(e => au.addEventListener(e, draw));
cv.addEventListener('click', ev => {
  const r = cv.getBoundingClientRect();
  au.currentTime = (ev.clientX - r.left) / r.width * total;
  draw();
});
window.addEventListener('resize', resize);
resize();
</script>
"""


def player_html(item):
    b64 = base64.b64encode(item["wav"]).decode()
    return (PLAYER.replace("__AUDIO__", b64)
                  .replace("__NOTES__", json.dumps(item["notes"]))
                  .replace("__TOTAL__", str(round(item["seconds"], 3))))


def render_player(html, height):
    html = html.strip()
    if hasattr(st, "iframe"):
        st.iframe(html, width="stretch", height=height)
    else:
        components.html(html, height=height)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------- header (shown immediately)
st.markdown("""
<div class="hero">
  <h1>🎼 AI Music Composer</h1>
  <p>A neural network (LSTM) that learned from classical MIDI music writes new pieces for you.</p>
  <span class="chip">🧠 PyTorch LSTM</span><span class="chip">🎹 4 instruments</span>
  <span class="chip">✍️ type your own start notes</span><span class="chip">📂 continue your own MIDI</span>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

# ---------------------------------------------------------------- studio (all settings in one form)
with st.sidebar:
    st.markdown("### 🎛️ Studio")
    with st.form("studio"):
        st.markdown("**Your start (optional)**")
        typed = st.text_input("Type your own opening notes", placeholder="C4 E4 G4 C5",
                              help="Space between notes. Use + for a chord (C4+E4+G4) and :0.5 for a short note (C4:0.5). "
                                   "The AI continues after your notes.")
        upload = st.file_uploader("...or upload a MIDI file", type=["mid", "midi"])
        st.markdown("**Sound and style**")
        mood = st.select_slider("Mood", options=list(MOODS), value="Balanced",
                                help="Calm = safe and regular, Creative = more surprising")
        instrument = st.selectbox("Instrument", list(INSTRUMENTS))
        key_name = st.selectbox("Key", KEYS, index=0,
                                help="The model writes in C major / A minor; this moves the piece. Ignored when you give your own start.")
        length = st.slider("Length (chords/notes the AI writes)", 40, 300, 100, step=10)
        bpm = st.slider("Tempo (BPM)", 60, 160, 100)
        versions = st.slider("Variations", 1, 3, 1, help="How many different pieces to compose at once")
        reverb = st.slider("Room reverb", 0, 50, 20, format="%d%%")
        with st.expander("Advanced sampling"):
            custom = st.checkbox("Use the values below instead of the mood")
            temperature = st.slider("Temperature", 0.3, 1.5, 0.9, 0.05)
            top_k = st.slider("Top-k", 1, 30, 8)
            top_p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.05)
            avoid_loops = st.checkbox("Avoid exact loops", help="The same short run of notes cannot appear twice in a piece")
            seed = st.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
        go = st.form_submit_button("🎹 Compose")
    st.caption("Nothing happens until you press **Compose**, so the music keeps playing while you change settings.")


# ---------------------------------------------------------------- the neural network (loaded once, after the page is visible)
@st.cache_resource(show_spinner="Loading the neural network (only the first time)...")
def get_engine():
    import generator  # imports PyTorch, which is slow, so it is done only here
    model, vocab = generator.load_model(MODEL_PATH)
    return generator, model, vocab, generator.load_seeds(DATA_DIR)


engine, model, vocab, seeds = get_engine()
st.session_state.setdefault("batches", [])
st.session_state.setdefault("runs", 0)

if go:
    try:
        if not custom:
            temperature, top_k, top_p = MOODS[mood]
        primer_ids, primer_note = None, ""
        if typed.strip():
            primer_ids, primer_note = engine.primer_from_notes(typed, vocab), "starts with your notes"
        elif upload is not None:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                tmp.write(upload.read())
            primer_ids, primer_note = engine.primer_from_midi(tmp.name, vocab), "continues your MIDI"
        shift = 0
        if primer_ids is None:  # with your own start the key stays as you wrote it
            shift = KEYS.index(key_name)
            shift = shift if shift <= 6 else shift - 12
        os.makedirs(OUT_DIR, exist_ok=True)
        st.session_state.runs += 1
        run_id = f"{time.strftime('%H%M%S')}_{st.session_state.runs}"  # unique even if you press Compose twice in a second
        clock, items = time.strftime("%H:%M:%S"), []
        began = time.time()
        with st.status("Composing...", expanded=True) as status:
            for v in range(versions):
                tag = f"Version {v + 1}: " if versions > 1 else ""
                if seed:
                    engine.set_seed(int(seed) + v)
                st.write(f"🧠 {tag}the neural network is writing the notes...")
                t0 = time.time()
                start = primer_ids or engine.pick_seed(seeds, vocab)
                written = engine.generate_tokens(model, vocab, start, length, temperature, top_k, top_p,
                                                 no_repeat=16 if avoid_loops else 0)
                t_network = time.time() - t0
                stats = analyse(written)  # measured in the original key (C / Am), only the AI's own notes
                tokens = ([vocab[i] for i in primer_ids] if primer_ids else []) + written
                tokens = transpose_tokens(tokens, shift)
                st.write(f"🎹 {tag}creating the sound ({instrument})...")
                name = os.path.join(OUT_DIR, f"app_{run_id}_{v + 1}")
                tokens_to_midi(tokens, name + ".mid", bpm, instrument)
                tokens_to_wav(tokens, name + ".wav", bpm, reverb=reverb / 100, instrument=instrument)
                notes, seconds = tokens_to_notes(tokens, bpm)
                items.append({
                    "wav_path": os.path.abspath(name + ".wav"), "wav": open(name + ".wav", "rb").read(),
                    "mid": open(name + ".mid", "rb").read(), "notes": notes, "seconds": seconds,
                    "n_notes": len(notes), "tokens": tokens, "t_network": t_network,
                    "in_key": stats["scale_ratio"] if stats else 0.0,
                    "variety": stats["distinct_4gram"] if stats else 0.0,
                    "label": " - ".join(x for x in [instrument, mood if not custom else "custom",
                                                    f"key of {key_name}" if not primer_ids else "", f"{bpm} BPM", primer_note] if x),
                })
            total = time.time() - began
            status.update(label=f"Done in {total:.1f} s", state="complete", expanded=False)
        st.session_state.batches.insert(0, {"id": run_id, "clock": clock, "items": items, "total": total})
        del st.session_state.batches[MAX_BATCHES:]
    except Exception as error:
        st.error(f"Could not generate music: {error}")


def show_piece(item, uid):
    render_player(player_html(item), 335)
    c = st.columns(4)
    c[0].metric("Duration", f"{item['seconds']:.0f} s")
    c[1].metric("Notes", item["n_notes"])
    c[2].metric("Notes in key", f"{item['in_key']:.0%}")
    c[3].metric("Variety", f"{item['variety']:.0%}")
    d = st.columns(3)
    d[0].download_button("⬇ MIDI file", item["mid"], f"ai_music_{uid}.mid", "audio/midi", key=f"mid_{uid}")
    d[1].download_button("⬇ WAV audio", item["wav"], f"ai_music_{uid}.wav", "audio/wav", key=f"wav_{uid}")
    if winsound is not None:
        if d[2].button("🔊 Play on this PC", key=f"play_{uid}", help="Plays through Windows, not the browser"):
            winsound.PlaySound(item["wav_path"], winsound.SND_FILENAME | winsound.SND_ASYNC)
        if d[2].button("⏹ Stop", key=f"stop_{uid}"):
            winsound.PlaySound(None, winsound.SND_PURGE)
    st.caption(f"The neural network wrote these notes in {item['t_network']:.1f} s. "
               "First tokens: " + " ".join(item["tokens"][:24]) + " ...")


tab_compose, tab_model, tab_about = st.tabs(["🎹 Compose", "📊 Model & results", "ℹ️ How it works"])

with tab_compose:
    if not st.session_state.batches:
        st.info("Choose your settings in the **Studio** on the left and press **Compose**.")
        a, b, c = st.columns(3)
        a.markdown('<div class="card"><h4>1 · Give a start (or not)</h4><p>Type a few notes like '
                   '<b>C4 E4 G4</b>, upload a MIDI file, or let the AI start from scratch.</p></div>', unsafe_allow_html=True)
        b.markdown('<div class="card"><h4>2 · The AI composes</h4><p>The LSTM writes new notes one by one. '
                   'Every press on Compose gives a different piece.</p></div>', unsafe_allow_html=True)
        c.markdown('<div class="card"><h4>3 · Listen and download</h4><p>Watch the notes light up while the music '
                   'plays, then save the MIDI or WAV.</p></div>', unsafe_allow_html=True)
    for n, batch in enumerate(st.session_state.batches):
        bid = batch["id"]
        st.subheader("Latest composition" if n == 0 else f"Earlier composition ({batch['clock']})")
        items = batch["items"]
        if len(items) == 1:
            st.markdown(f"**{items[0]['label']}**")
            show_piece(items[0], f"{bid}_1")
        else:
            for i, tab in enumerate(st.tabs([f"Version {i + 1}" for i in range(len(items))])):
                with tab:
                    st.markdown(f"**{items[i]['label']}**")
                    show_piece(items[i], f"{bid}_{i + 1}")
        st.divider()

with tab_model:
    history, report = read_json("model/history.json"), read_json("evaluation/report.json")
    params = sum(p.numel() for p in model.parameters())
    if history:
        best = min(history, key=lambda h: h["val_loss"])
        m = st.columns(5)
        m[0].metric("Parameters", f"{params / 1e6:.2f} M")
        m[1].metric("Vocabulary", f"{len(vocab)} tokens")
        m[2].metric("Epochs trained", len(history))
        m[3].metric("Validation accuracy", f"{best['val_acc']:.1%}")
        m[4].metric("Perplexity", f"{best['val_perplexity']:.1f}")
    else:
        st.info("Train the model (`python train.py`) to see the training results here.")
    if os.path.exists("model/training_curve.png"):
        st.markdown("##### Training curve")
        st.image("model/training_curve.png")
    if report:
        st.markdown("##### Generated music vs real Bach music (never seen by the model)")
        labels = {"scale_ratio": "Notes inside the key", "distinct_4gram": "Variety (distinct 4-grams)",
                  "rest_ratio": "Share of rests", "avg_polyphony": "Notes per chord",
                  "pitch_range": "Pitch range (semitones)", "avg_leap": "Average melodic jump",
                  "avg_note_length": "Average note length (beats)"}
        st.table([{"Metric": text, "Real music": round(report["real"][k], 2), "Generated": round(report["generated"][k], 2)}
                  for k, text in labels.items()])
        st.metric("Pitch-class similarity", f"{report['pitch_class_similarity']:.2f}",
                  help="1.00 = the generated music uses the notes exactly like real music")
    if os.path.exists("evaluation/evaluation.png"):
        st.image("evaluation/evaluation.png")
    elif not report:
        st.caption("Run `python evaluate.py` to compare generated and real music here.")

with tab_about:
    st.markdown("""
**Who writes the music?** The neural network does. When you press *Compose*, the LSTM predicts the next note again and again
(with a bit of randomness), so every piece is new and nothing is stored in advance. The built-in synthesizer only turns the notes
into sound, like a pianist reading sheet music. You steer it with a start, a mood, an instrument, a key and a tempo.

**The pipeline**

1. **Collect:** classical MIDI files (Bach and others) from the music21 corpus.
2. **Preprocess:** every song is moved to C major / A minor and turned into tokens. A chord becomes one token per note
   plus one token for its length, for example `N48 N64 N67 D0.5` = three notes together for half a beat.
   The whole vocabulary is only about 70 tokens.
3. **Train:** an LSTM network (embedding, 2 LSTM layers, dropout) learns to guess the next token.
   Whole songs are kept apart for validation, so the scores are honest.
4. **Generate:** the model reads a start (a real song opening, your typed notes or your MIDI) and continues it token by token.
   *Temperature*, *top-k* and *top-p* decide how safe or surprising each next note is.
5. **Evaluate:** generated music is compared with real music using numbers (notes in key, variety, rests, ...).

**Good to know:** the model is small and trained on Bach, so the result is Bach-style music with dense four-note chords,
not a finished song. A higher *Creative* mood gives more variety but also more wrong notes.
""")
