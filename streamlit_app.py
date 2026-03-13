import streamlit as st
from google import genai
import pdfplumber
import json
import io
import re
import requests
import traceback

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Regulus Engine — Nuclear Demo",
    page_icon="☢",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background: #0b1120; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.stApp { background: #0b1120; color: #e8edf5; }
.block-container { padding-top: 2rem; max-width: 1200px; }

div[data-testid="stFileUploader"] {
    border: 2px dashed #1e3050;
    border-radius: 10px;
    padding: 1rem;
    background: #0f1a2e;
}

.card {
    background: #0f1a2e;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-en  { border-top: 3px solid #ff6b9d; }
.card-bm  { border-top: 3px solid #00e4b8; }

.tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}
.tag-en      { background: rgba(255,107,157,0.15);  color: #ff6b9d; border: 1px solid rgba(255,107,157,0.3); }
.tag-bm      { background: rgba(0,228,184,0.15);   color: #00e4b8; border: 1px solid rgba(0,228,184,0.3); }
.tag-sealion { background: rgba(255,179,71,0.15);  color: #ffb347; border: 1px solid rgba(255,179,71,0.3); }

.field-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #7a8ba8;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.field-value {
    font-size: 0.9rem;
    color: #e8edf5;
    margin-bottom: 1rem;
    line-height: 1.6;
}

.task-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 10px 12px;
    border-left: 2px solid #ffb347;
    background: rgba(255,179,71,0.04);
    border-radius: 0 6px 6px 0;
    margin-bottom: 8px;
}
.task-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #ffb347;
    min-width: 22px;
    padding-top: 2px;
}
.task-text { font-size: 0.85rem; color: #e8edf5; line-height: 1.55; }

.warning-row {
    padding: 8px 12px;
    border-left: 2px solid #ff6b9d;
    background: rgba(255,107,157,0.04);
    border-radius: 0 4px 4px 0;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #e8edf5;
}

.caution-row {
    padding: 8px 12px;
    border-left: 2px solid #ffb347;
    background: rgba(255,179,71,0.04);
    border-radius: 0 4px 4px 0;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #e8edf5;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #7a8ba8;
    margin-bottom: 4px;
}
.step-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

.radiation-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
}
.rad-high   { background: rgba(255,71,87,0.15); color: #ff4757; border: 1px solid rgba(255,71,87,0.3); }
.rad-medium { background: rgba(255,179,71,0.15); color: #ffb347; border: 1px solid rgba(255,179,71,0.3); }
.rad-low    { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
</style>
""", unsafe_allow_html=True)

# ── Model config ──────────────────────────────────────────────────────────────

GEMINI_MODEL  = "gemini-2.5-flash"
SEALION_MODEL = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
SEALION_BASE  = "https://api.sea-lion.ai/v1"

MAX_CHARS = 12_000

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_parse_json(raw: str) -> dict:
    """Robustly extract JSON from LLM output that may contain markdown fences or preamble."""
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract valid JSON from LLM response", cleaned, 0)


def esc(text):
    """Escape HTML entities."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── API clients ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_gemini():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def sealion_translate(english_card: dict, target_lang: str, api_key: str) -> dict:
    lang_name  = "Bahasa Melayu" if target_lang == "bm" else "Bahasa Indonesia"
    authority  = "AELB/MOSTI" if target_lang == "bm" else "BAPETEN"

    prompt = f"""You are a certified nuclear safety translator specialising in {lang_name}.

Translate the following English nuclear safety procedure card into {lang_name} for {authority} compliance.

Rules:
- Return ONLY a valid JSON object — no preamble, no markdown fences
- Preserve ALL JSON key names exactly as-is (they stay in English)
- Translate all string values into {lang_name}
- Regulation numbers, equipment model numbers, and technical references stay in English
- Use correct {authority} regulatory terminology
- Units of measurement (Sv, mSv, Bq, kBq) stay in their international form

English safety procedure card:
{json.dumps(english_card, ensure_ascii=False, indent=2)}

Return ONLY the translated JSON object."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": SEALION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 3072,
    }

    resp = requests.post(
        f"{SEALION_BASE}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    return safe_parse_json(raw)


def gemini_translate(english_card: dict, target_lang: str) -> dict:
    client = get_gemini()
    lang_name  = "Bahasa Melayu" if target_lang == "bm" else "Bahasa Indonesia"
    authority  = "AELB/MOSTI" if target_lang == "bm" else "BAPETEN"

    prompt = f"""You are a certified nuclear safety translator specialising in {lang_name}.

Translate the following English nuclear safety procedure card into {lang_name} for {authority} compliance.

Rules:
- Return ONLY a valid JSON object — no preamble, no markdown fences
- Preserve ALL JSON key names exactly as-is (they stay in English)
- Translate all string values into {lang_name}
- Regulation numbers, equipment model numbers, and technical references stay in English
- Use correct {authority} regulatory terminology
- Units of measurement (Sv, mSv, Bq, kBq) stay in their international form

English safety procedure card:
{json.dumps(english_card, ensure_ascii=False, indent=2)}

Return ONLY the translated JSON object."""

    result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return safe_parse_json(result.text)


# ── Core generation (Gemini) ──────────────────────────────────────────────────

SAFETY_PROCEDURE_SCHEMA = """{
  "procedureNumber": "SPC-[document ref]-[sequential]",
  "documentReference": "Source regulation or document ID",
  "title": "Short procedure title",
  "facilityType": "e.g. Nuclear Power Plant / Research Reactor / Fuel Processing / Waste Storage",
  "issuingAuthority": "e.g. NRC / IAEA / BAPETEN / AELB",
  "applicability": "Which systems, equipment, or personnel this applies to",
  "frequencyOrTrigger": "e.g. Daily / Weekly / Before maintenance / After incident",
  "radiationZone": "GREEN / AMBER / RED",
  "maxPermittedDose": "e.g. 20 mSv/year or as specified",
  "requiredPPE": ["list of required PPE items"],
  "requiredEquipment": ["list of instruments or tools needed"],
  "prerequisites": ["conditions that must be met before starting"],
  "steps": [
    {
      "step": 1,
      "action": "What to do",
      "acceptanceCriteria": "How to verify this step is complete/correct",
      "reference": "Regulation or manual section if applicable"
    }
  ],
  "warnings": ["DANGER/WARNING statements"],
  "cautions": ["CAUTION statements"],
  "recordsRequired": ["What documentation must be completed"],
  "signoffRoles": ["Roles that must sign off"]
}"""


def generate_english_procedure(doc_text: str, doc_type: str) -> dict:
    client = get_gemini()
    snippet = doc_text[:MAX_CHARS]

    prompt = f"""You are a nuclear safety compliance expert with deep knowledge of IAEA safety standards, NRC regulations (10 CFR), and Southeast Asian nuclear regulatory frameworks (BAPETEN, AELB/MOSTI, OAP, VARANS).

Read the following nuclear regulatory document excerpt and generate a structured safety procedure card in English.

Your output must be precise and suitable for licensed nuclear facility operators. Never hallucinate regulation numbers — if a specific regulation is not mentioned in the source text, leave the reference field empty.

For radiation zones: GREEN = general area, AMBER = controlled area with dosimetry, RED = restricted/exclusion zone.

Document type: {doc_type}

Document text:
{snippet}

Return ONLY a valid JSON object matching this schema:
{SAFETY_PROCEDURE_SCHEMA}

Extract real regulation references from the text. Steps should be concrete and actionable. Include all relevant warnings and cautions.

Return ONLY the JSON object. No preamble, no markdown fences."""

    result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return safe_parse_json(result.text)


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF using getvalue() to avoid file position issues."""
    file_bytes = uploaded_file.getvalue()
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages).strip()


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_procedure_card(card: dict, lang: str, translation_engine: str = "gemini"):
    is_en = lang == "en"
    card_class = "card-en" if is_en else "card-bm"

    if is_en:
        tag_class = "tag-en"
        tag_label = "ENGLISH · GEMINI 2.5 FLASH"
    elif translation_engine == "sealion":
        tag_class = "tag-sealion"
        lang_str = "BAHASA MELAYU" if lang == "bm" else "BAHASA INDONESIA"
        tag_label = f"{lang_str} · SEA-LION v4 27B"
    else:
        tag_class = "tag-bm"
        lang_str = "BAHASA MELAYU" if lang == "bm" else "BAHASA INDONESIA"
        tag_label = f"{lang_str} · GEMINI 2.5 FLASH"

    html = f'<div class="card {card_class}"><span class="tag {tag_class}">{tag_label}</span>'

    def field(label, value):
        if not value:
            return ""
        return f'<div class="field-label">{label}</div><div class="field-value">{esc(value)}</div>'

    html += field("Procedure Number",    card.get("procedureNumber"))
    html += field("Document Reference",  card.get("documentReference"))
    html += field("Title",               card.get("title"))
    html += field("Facility Type",       card.get("facilityType"))
    html += field("Issuing Authority",   card.get("issuingAuthority"))
    html += field("Applicability",       card.get("applicability"))
    html += field("Frequency / Trigger", card.get("frequencyOrTrigger"))

    # Radiation zone badge
    zone = str(card.get("radiationZone", "")).upper()
    if zone:
        zone_class = "rad-high" if "RED" in zone else ("rad-medium" if "AMBER" in zone else "rad-low")
        html += f'<div class="field-label">RADIATION ZONE</div>'
        html += f'<div class="field-value"><span class="radiation-badge {zone_class}">☢ {esc(zone)}</span></div>'

    html += field("Max Permitted Dose",  card.get("maxPermittedDose"))

    ppe = card.get("requiredPPE") or []
    if ppe and isinstance(ppe, list):
        html += field("Required PPE", " · ".join(esc(p) for p in ppe))

    equip = card.get("requiredEquipment") or []
    if equip and isinstance(equip, list):
        html += field("Required Equipment", " · ".join(esc(e) for e in equip))

    prereqs = card.get("prerequisites") or []
    if prereqs and isinstance(prereqs, list):
        html += '<div class="field-label" style="margin-bottom:8px">PREREQUISITES</div>'
        for p in prereqs:
            html += f'<div style="font-size:0.85rem;color:#e8edf5;margin-bottom:4px;padding-left:12px">• {esc(p)}</div>'

    warnings = card.get("warnings") or []
    if warnings and isinstance(warnings, list):
        html += '<div class="field-label" style="margin-bottom:8px;margin-top:12px">⚠ WARNINGS</div>'
        for w in warnings:
            html += f'<div class="warning-row">☢ {esc(w)}</div>'

    cautions = card.get("cautions") or []
    if cautions and isinstance(cautions, list):
        html += '<div class="field-label" style="margin-bottom:8px;margin-top:12px">CAUTIONS</div>'
        for c in cautions:
            html += f'<div class="caution-row">⚠ {esc(c)}</div>'

    steps = card.get("steps") or []
    if steps and isinstance(steps, list):
        html += '<div class="field-label" style="margin-bottom:8px;margin-top:12px">PROCEDURE STEPS</div>'
        for s in steps:
            if not isinstance(s, dict):
                continue
            step_num = str(s.get("step", "")).zfill(2)
            action = esc(s.get("action", ""))
            criteria = esc(s.get("acceptanceCriteria", "") or "")
            ref = esc(s.get("reference", "") or "")
            criteria_html = f'<div style="font-size:0.75rem;color:#34d399;margin-top:3px">✓ {criteria}</div>' if criteria else ""
            ref_html = f'<div style="font-size:0.75rem;color:#7a8ba8;margin-top:2px">Ref: {ref}</div>' if ref else ""
            html += f'<div class="task-row"><span class="task-num">{step_num}</span><div class="task-text">{action}{criteria_html}{ref_html}</div></div>'

    records = card.get("recordsRequired") or []
    if records and isinstance(records, list):
        html += field("Records Required", " · ".join(esc(r) for r in records))

    signoff = card.get("signoffRoles") or []
    if signoff and isinstance(signoff, list):
        html += field("Sign-off Required", " · ".join(esc(s) for s in signoff))

    html += "</div>"
    return html


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 2rem">
  <div style="font-family: Space Mono, monospace; font-size: 0.65rem; letter-spacing: 0.25em; color: #00e4b8; font-weight: 700; margin-bottom: 6px">
    REGULUS ENGINE — NUCLEAR DEMO
  </div>
  <h1 style="font-size: 1.75rem; font-weight: 400; color: #e8edf5; margin: 0">
    Nuclear Regulatory Doc → Safety Procedure Card
  </h1>
  <p style="color: #7a8ba8; font-size: 0.85rem; margin-top: 8px">
    Upload a nuclear regulatory document (PDF). Gemini generates the safety procedure card.<br>
    Choose a translation engine below to produce the bilingual version.
  </p>
</div>
""", unsafe_allow_html=True)

mode = st.radio(
    "Translation pipeline",
    options=["gemini", "sealion"],
    format_func=lambda x: (
        "Gemini 2.5 Flash — translation"
        if x == "gemini"
        else "SEA-LION v4 27B — translation (regional specialist)"
    ),
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "gemini":
    pipeline_html = """
    <div class="card" style="margin:1rem 0 1.5rem; display:flex; gap:2rem; flex-wrap:wrap; align-items:center">
      <div class="pipeline-step"><div class="step-dot" style="background:#4d9fff"></div>PDF Upload</div>
      <div style="color:#1e3050; font-family:monospace">→</div>
      <div class="pipeline-step"><div class="step-dot" style="background:#4d9fff"></div>Text Extraction</div>
      <div style="color:#1e3050; font-family:monospace">→</div>
      <div class="pipeline-step"><div class="step-dot" style="background:#9d7aff"></div>Procedure Generation (Gemini 2.5 Flash)</div>
      <div style="color:#1e3050; font-family:monospace">→</div>
      <div class="pipeline-step"><div class="step-dot" style="background:#9d7aff"></div>Translation (Gemini 2.5 Flash)</div>
    </div>"""
else:
    pipeline_html = """
    <div class="card" style="margin:1rem 0 1.5rem; display:flex; gap:2rem; flex-wrap:wrap; align-items:center">
      <div class="pipeline-step"><div class="step-dot" style="background:#4d9fff"></div>PDF Upload</div>
      <div style="color:#1e3050; font-family:monospace">→</div>
      <div class="pipeline-step"><div class="step-dot" style="background:#4d9fff"></div>Text Extraction</div>
      <div style="color:#1e3050; font-family:monospace">→</div>
      <div class="pipeline-step"><div class="step-dot" style="background:#9d7aff"></div>Procedure Generation (Gemini 2.5 Flash)</div>
      <div style="color:#1e3050; font-family:monospace">→</div>
      <div class="pipeline-step"><div class="step-dot" style="background:#ffb347"></div>Translation (SEA-LION v4 27B)</div>
    </div>"""

st.markdown(pipeline_html, unsafe_allow_html=True)

col_upload, col_options = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload nuclear regulatory document (PDF)",
        type=["pdf"],
        label_visibility="collapsed",
    )

with col_options:
    doc_type = st.selectbox(
        "Document type",
        options=[
            "IAEA Safety Standard",
            "NRC Regulation (10 CFR)",
            "BAPETEN Regulation (Indonesia)",
            "AELB/MOSTI Regulation (Malaysia)",
            "OAP Regulation (Thailand)",
            "VARANS Regulation (Vietnam)",
            "Nuclear Facility Operating Procedure",
            "Radiation Protection Manual",
            "Other / Unknown",
        ],
    )
    target_lang = st.selectbox(
        "Translation language",
        options=["bm", "bi"],
        format_func=lambda x: "🇲🇾 Bahasa Melayu" if x == "bm" else "🇮🇩 Bahasa Indonesia",
    )
    show_raw   = st.checkbox("Show extracted PDF text", value=False)
    show_json  = st.checkbox("Show raw JSON", value=False)
    debug_mode = st.checkbox("Debug mode", value=False)

st.markdown(f"""
<div class="card" style="border-top: 2px solid #4d9fff; padding: 12px 16px;">
  <div style="font-family: Space Mono, monospace; font-size: 0.6rem; letter-spacing: 0.1em; color: #4d9fff; font-weight: 700; margin-bottom: 4px">
    TRUNCATION NOTE
  </div>
  <div style="font-size: 0.8rem; color: #7a8ba8; line-height: 1.6">
    This demo truncates documents to the first <strong style="color:#e8edf5">{MAX_CHARS:,}</strong> characters before sending to Gemini.
    For 100+ page regulatory documents, this covers roughly the first 5–8 pages. The full MVP uses RAG with chunked semantic search.
  </div>
</div>
""", unsafe_allow_html=True)


# ── Process ───────────────────────────────────────────────────────────────────

if uploaded:
    # Step 0: Extract text
    with st.spinner("Extracting text from PDF…"):
        try:
            doc_text = extract_text_from_pdf(uploaded)
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            if debug_mode:
                st.code(traceback.format_exc())
            st.stop()

    if not doc_text.strip():
        st.error("Could not extract text from this PDF. It may be scanned/image-only.")
        st.stop()

    char_count = len(doc_text)
    page_est = max(1, char_count // 1800)

    st.markdown(f"""
    <div style="font-family: Space Mono, monospace; font-size: 0.7rem; color: #7a8ba8; margin: 8px 0 16px">
      Extracted {char_count:,} characters (~{page_est} pages).
      {"Truncating to first " + f"{MAX_CHARS:,}" + " chars." if char_count > MAX_CHARS else "Full text will be used."}
    </div>
    """, unsafe_allow_html=True)

    if show_raw:
        with st.expander("Extracted PDF text"):
            st.text(doc_text[:8000] + ("…" if len(doc_text) > 8000 else ""))

    # Step 1: Generate with Gemini
    with st.spinner("Generating safety procedure card with Gemini 2.5 Flash…"):
        try:
            english_card = generate_english_procedure(doc_text, doc_type)
        except json.JSONDecodeError as e:
            st.error("Gemini returned invalid JSON for generation. Toggle debug mode for details.")
            if debug_mode:
                st.code(traceback.format_exc())
            st.stop()
        except Exception as e:
            st.error(f"Generation error: {type(e).__name__}: {e}")
            if debug_mode:
                st.code(traceback.format_exc())
            st.stop()

    if debug_mode:
        with st.expander("Debug: English card (parsed)"):
            st.json(english_card)

    # Step 2: Translate
    if mode == "gemini":
        lang_label = "Bahasa Melayu" if target_lang == "bm" else "Bahasa Indonesia"
        with st.spinner(f"Translating into {lang_label} with Gemini 2.5 Flash…"):
            try:
                translated_card = gemini_translate(english_card, target_lang)
            except json.JSONDecodeError:
                st.error("Gemini returned invalid JSON. Toggle debug mode for details.")
                if debug_mode:
                    st.code(traceback.format_exc())
                st.stop()
            except Exception as e:
                st.error(f"Gemini error: {type(e).__name__}: {e}")
                if debug_mode:
                    st.code(traceback.format_exc())
                st.stop()
        translation_engine = "gemini"
    else:
        sealion_key = st.secrets.get("SEALION_API_KEY", "")
        if not sealion_key:
            st.error("SEA-LION API key not found. Add `SEALION_API_KEY` to your Streamlit secrets.")
            st.stop()

        lang_label = "Bahasa Melayu" if target_lang == "bm" else "Bahasa Indonesia"
        with st.spinner(f"Translating into {lang_label} with SEA-LION v4 27B…"):
            try:
                translated_card = sealion_translate(english_card, target_lang, sealion_key)
            except json.JSONDecodeError:
                st.error("SEA-LION returned invalid JSON. Toggle debug mode for details.")
                if debug_mode:
                    st.code(traceback.format_exc())
                st.stop()
            except requests.HTTPError as e:
                st.error(f"SEA-LION API error: {e}")
                if debug_mode:
                    st.code(traceback.format_exc())
                st.stop()
            except Exception as e:
                st.error(f"SEA-LION error: {type(e).__name__}: {e}")
                if debug_mode:
                    st.code(traceback.format_exc())
                st.stop()
        translation_engine = "sealion"

    if debug_mode:
        with st.expander("Debug: Translated card (parsed)"):
            st.json(translated_card)

    # Output
    st.markdown("---")
    col_en, col_tr = st.columns(2)

    with col_en:
        st.markdown(render_procedure_card(english_card, "en"), unsafe_allow_html=True)
        if show_json:
            with st.expander("English JSON"):
                st.json(english_card)

    with col_tr:
        st.markdown(
            render_procedure_card(translated_card, target_lang, translation_engine),
            unsafe_allow_html=True,
        )
        if show_json:
            lang_label = "Bahasa Melayu" if target_lang == "bm" else "Bahasa Indonesia"
            with st.expander(f"{lang_label} JSON"):
                st.json(translated_card)

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem; border-style: dashed;">
      <div style="font-size: 2rem; margin-bottom: 12px">☢</div>
      <div style="color: #7a8ba8; font-size: 0.9rem">Upload a nuclear regulatory PDF above to generate a bilingual safety procedure card</div>
      <div style="color: #4a5568; font-size: 0.75rem; margin-top: 12px">
        Supports: IAEA Safety Standards · NRC 10 CFR · BAPETEN · AELB/MOSTI · OAP · VARANS
      </div>
    </div>
    """, unsafe_allow_html=True)