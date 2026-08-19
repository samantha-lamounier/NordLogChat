import os
import re
import json
from pathlib import Path
import numpy as np
import streamlit as st
import voyageai
from anthropic import Anthropic

st.set_page_config(page_title="NordLog Ops Copilot", page_icon="🚚", layout="wide")

# ---------- Design tokens (clean chat-app look, inspired by familiar LLM UIs) ----------
BG = "#FAF9F5"
SURFACE = "#FFFFFF"
BORDER = "#E8E4DB"
USER_BUBBLE = "#F0EEE6"
TEXT = "#262521"
MUTED = "#8A867C"
ACCENT = "#CC785C"      # terracotta — used for send button + "doc search" tag
ACCENT_2 = "#5B7B8C"    # muted slate blue — "db query" tag

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

.stApp {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Inter', sans-serif;
}}
.mono {{ font-family: 'IBM Plex Mono', monospace; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ---- header ---- */
.brand-title {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
.brand-sub {{ font-size: 12px; color: {MUTED}; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 2px; }}
.brand-tag {{ font-size: 12px; color: {MUTED}; text-align: right; line-height: 1.5; }}

/* ---- chat messages, no boxy bubbles for the agent ---- */
.chat-row {{ display: flex; margin-bottom: 22px; }}
.chat-row.user {{ justify-content: flex-end; }}
.chat-row.agent {{ justify-content: flex-start; align-items: flex-start; gap: 10px; }}
.avatar {{
    width: 26px; height: 26px; border-radius: 50%;
    background: {ACCENT}; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
}}
.bubble-user {{
    background: {USER_BUBBLE}; border-radius: 18px; padding: 10px 16px;
    max-width: 70%; font-size: 15px; line-height: 1.55; color: {TEXT};
}}
.agent-text {{ max-width: 78%; font-size: 15px; line-height: 1.6; padding-top: 3px; color: {TEXT}; }}

/* ---- suggestion "chips" as plain text, not boxes ---- */
div[data-testid="stHorizontalBlock"] .stButton > button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: {MUTED} !important;
    font-size: 13px !important;
    text-align: left !important;
    padding: 4px 0 !important;
    white-space: normal !important;
    height: auto !important;
}}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {{
    color: {ACCENT} !important;
    text-decoration: underline;
}}

/* ---- input pill ---- */
div[data-testid="stForm"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 26px !important;
    padding: 6px 6px 6px 20px !important;
    background: {SURFACE} !important;
}}
div[data-testid="stForm"] input {{
    border: none !important;
    background: transparent !important;
    font-size: 15px !important;
    box-shadow: none !important;
}}
div[data-testid="stForm"] .stButton > button {{
    background: {ACCENT} !important;
    color: #fff !important;
    border-radius: 50% !important;
    width: 38px !important; height: 38px !important;
    padding: 0 !important;
    font-size: 16px !important;
    box-shadow: none !important;
}}
div[data-testid="stForm"] .stButton > button:hover {{ opacity: 0.9; }}

/* ---- trace panel ---- */
.trace-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
}}
.badge {{ font-size: 10px; letter-spacing: 1px; font-weight: 600; text-transform: uppercase; }}
.record {{
    background: #F5F3EC;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 12px;
    line-height: 1.5;
}}
[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ---------- Data (loaded from external files, not hardcoded) ----------
DATA_DIR = Path(__file__).parent / "data"
ORDERS = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))
DOCS = json.loads((DATA_DIR / "docs.json").read_text(encoding="utf-8"))

SUGGESTIONS = [
    "Onde está o pedido NL-52210 e por que ele está atrasado?",
    "Qual é a nossa política de sinistro por avaria?",
    "O pedido NL-48213 está retido num posto fiscal — explica por que e o que dizemos ao cliente.",
]

SYSTEM_PROMPT = """You are the routing brain behind NordLog's internal Ops Copilot. You receive a support question, NordLog's full order database, and a short list of policy document CANDIDATES that a vector search has already retrieved as the most semantically similar to the question (each candidate includes its similarity score). Decide whether answering requires the ORDER DATABASE (structured facts: status, ETA, carrier, region, delay reason), the POLICY CANDIDATES, or BOTH.
Write the "answer" field in Brazilian Portuguese (PT-BR), regardless of the language the question was asked in.
Respond with strict JSON only — no markdown fences, no prose outside the JSON — matching exactly this shape:
{"route": "database" | "semantic" | "both", "matched_order_ids": string[], "matched_doc_ids": string[], "answer": string}
Only include a doc id in matched_doc_ids if it is actually relevant to the question and grounds part of the answer — a high similarity score alone does not guarantee relevance, use judgment. The answer must be 2-4 sentences, written like a sharp ops teammate, citing concrete details (order id, status, eta, or exact policy terms) pulled only from matched records. If nothing matches, say so plainly instead of guessing."""

VOYAGE_MODEL = "voyage-3.5"
TOP_K_DOCS = 3


@st.cache_resource(show_spinner="Gerando embeddings da base de documentos…")
def embed_doc_library(voyage_api_key: str, docs_json: str) -> np.ndarray:
    """Embeds every policy doc once per session/key. Cached so it only runs on the first question."""
    vo = voyageai.Client(api_key=voyage_api_key)
    docs = json.loads(docs_json)
    texts = [d["body"] for d in docs]
    result = vo.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return np.array(result.embeddings)


def semantic_retrieve(question: str, voyage_api_key: str, doc_embeddings: np.ndarray):
    """Embeds the question and ranks docs by cosine similarity — the actual semantic layer."""
    vo = voyageai.Client(api_key=voyage_api_key)
    q_embedding = np.array(vo.embed([question], model=VOYAGE_MODEL, input_type="query").embeddings[0])
    sims = doc_embeddings @ q_embedding / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_embedding)
    )
    ranked_idx = np.argsort(-sims)[:TOP_K_DOCS]
    return [{"doc": DOCS[i], "score": float(sims[i])} for i in ranked_idx]


def ask_copilot(question: str, anthropic_client: Anthropic, voyage_api_key: str, doc_embeddings: np.ndarray) -> dict:
    candidates = semantic_retrieve(question, voyage_api_key, doc_embeddings)
    candidate_payload = [
        {"id": c["doc"]["id"], "title": c["doc"]["title"], "body": c["doc"]["body"], "similarity_score": round(c["score"], 4)}
        for c in candidates
    ]

    response = anthropic_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps({"question": question, "orders": ORDERS, "policy_candidates": candidate_payload})}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    clean = re.sub(r"```json|```", "", text).strip()
    result = json.loads(clean)
    result["retrieval_scores"] = {c["doc"]["id"]: c["score"] for c in candidates}
    return result


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "agent", "text": "NordLog Ops Copilot online. Pergunte sobre um pedido ou uma política — eu busco na fonte certa."}
    ]
if "trace" not in st.session_state:
    st.session_state.trace = []

# ---------- Sidebar ----------
api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
voyage_key = st.secrets.get("VOYAGE_API_KEY", os.environ.get("VOYAGE_API_KEY", ""))

with st.sidebar:
    st.markdown("### NordLog Ops Copilot")
    if api_key and voyage_key:
        st.caption("✅ Conectado — chaves gerenciadas via Secrets do Streamlit Cloud.")
    else:
        missing = []
        if not api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not voyage_key:
            missing.append("VOYAGE_API_KEY")
        st.error(f"Faltando: {', '.join(missing)}. Adicione em Settings > Secrets no Streamlit Cloud.")

doc_embeddings = None
if voyage_key:
    doc_embeddings = embed_doc_library(voyage_key, json.dumps(DOCS))

# ---------- Header ----------
col_logo, col_tag = st.columns([3, 2])
with col_logo:
    st.markdown("<div class='brand-title'>NordLog</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-sub'>Ops Copilot · powered by Waypoint AI</div>", unsafe_allow_html=True)
with col_tag:
    st.markdown("<div class='brand-tag'>uma pergunta, duas fontes<br/>banco de dados + biblioteca de políticas</div>", unsafe_allow_html=True)
st.write("")

chat_col, trace_col = st.columns([1.3, 1])

# ---------- Chat column ----------
with chat_col:
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"<div class='chat-row user'><div class='bubble-user'>{m['text']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='chat-row agent'><div class='avatar'>🚚</div><div class='agent-text'>{m['text']}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    sugg_cols = st.columns(len(SUGGESTIONS))
    clicked = None
    for i, s in enumerate(SUGGESTIONS):
        if sugg_cols[i].button(s, key=f"sugg_{i}"):
            clicked = s

    with st.form("ask_form", clear_on_submit=True):
        c1, c2 = st.columns([10, 1])
        with c1:
            typed = st.text_input("pergunta", label_visibility="collapsed", placeholder="Pergunte sobre um pedido ou uma política…")
        with c2:
            submitted = st.form_submit_button("↑")

    question = clicked or (typed if submitted and typed else None)

    if question:
        if not api_key or not voyage_key:
            st.error("Adicione as duas chaves (Anthropic e Voyage AI) na barra lateral.")
        else:
            st.session_state.messages.append({"role": "user", "text": question})
            client = Anthropic(api_key=api_key)
            with st.spinner("gerando embedding da pergunta + roteando…"):
                try:
                    result = ask_copilot(question, client, voyage_key, doc_embeddings)
                    st.session_state.messages.append({"role": "agent", "text": result["answer"]})
                    matched_orders = [o for o in ORDERS if o["id"] in result.get("matched_order_ids", [])]
                    matched_docs = [d for d in DOCS if d["id"] in result.get("matched_doc_ids", [])]
                    st.session_state.trace.insert(0, {
                        "question": question,
                        "route": result.get("route", "unknown"),
                        "orders": matched_orders,
                        "docs": matched_docs,
                        "scores": result.get("retrieval_scores", {}),
                    })
                except Exception:
                    st.session_state.messages.append({"role": "agent", "text": "Erro ao consultar — tente reformular a pergunta."})
            st.rerun()

# ---------- Trace column ----------
with trace_col:
    st.markdown("<div style='font-weight:700;font-size:15px;margin-bottom:2px;'>Rastreio da resposta</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:{MUTED};margin-bottom:16px;'>como o agente respondeu cada pergunta</div>", unsafe_allow_html=True)

    if not st.session_state.trace:
        st.markdown(
            f"<div style='font-size:13px;color:{MUTED};border:1px dashed {BORDER};border-radius:12px;padding:16px;'>"
            "Faça uma pergunta para ver a decisão de roteamento rastreada aqui.</div>",
            unsafe_allow_html=True,
        )

    for t in st.session_state.trace:
        route = t["route"]
        if route == "database":
            badge_color, badge_label = ACCENT_2, "Consulta ao banco"
        elif route == "semantic":
            badge_color, badge_label = ACCENT, "Busca em política"
        else:
            badge_color, badge_label = TEXT, "Banco + busca em política"

        html = f"<div class='trace-card'><div class='badge' style='color:{badge_color};'>{badge_label}</div>"
        html += f"<div style='font-size:13px;color:{MUTED};margin:8px 0;'>{t['question']}</div>"
        for o in t["orders"]:
            html += f"<div class='record'><div style='color:{ACCENT_2};font-weight:600;'>{o['id']} — {o['customer']}</div>"
            html += f"<div style='color:{MUTED};margin-top:2px;'>{o['status']} · {o['carrier']} · ETA {o['eta']}</div>"
            if o.get("delay_reason"):
                html += f"<div style='color:{MUTED};margin-top:2px;'>{o['delay_reason']}</div>"
            html += "</div>"
        for d in t["docs"]:
            score = t.get("scores", {}).get(d["id"])
            score_html = f" <span class='mono' style='color:{MUTED};'>· similaridade {score:.3f}</span>" if score is not None else ""
            html += f"<div class='record'><div style='color:{ACCENT};font-weight:600;'>{d['title']}{score_html}</div>"
            html += f"<div style='color:{MUTED};margin-top:2px;'>{d['body']}</div></div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
