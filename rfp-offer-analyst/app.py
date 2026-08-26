import json

import streamlit as st

from chat import answer_offer_question
from db import get_db, init_db
from grade import run_grading_batch
from memory import get_tender_memory_notes, store_tender_memories
from utils import calculate_scores, extract, is_scanned, seed_if_empty
from vector_store import delete_offer_chunks, index_offer_chunks, is_offer_indexed

st.set_page_config(page_title="RFP Offer Analyst", layout="wide", initial_sidebar_state="expanded")

STYLE = """
<style>
:root {
  --paper: #EFEDE6;
  --surface: #FAF9F5;
  --white: #FFFFFF;
  --ink: #17160F;
  --ink2: #57554A;
  --ink3: #8B8878;
  --rule: #DAD6C8;
  --rule2: #C6C1AE;
  --accent: #A8442A;
  --accent-soft: #F3E2DC;
  --full: #2F7A4E;
  --full-bg: #DCEADF;
  --partial: #B07C12;
  --partial-bg: #F5E7C8;
  --none: #A93326;
  --none-bg: #F2DAD5;
  --unclear: #87846F;
  --unclear-bg: #E3E0D3;
}

/* App Background & Typography */
.stApp {
  background-color: var(--paper);
  color: var(--ink);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background-color: var(--ink) !important;
  border-right: 1px solid #2E2C22;
}

/* Sidebar Navigation Buttons - Dark Theme Overrides */
[data-testid="stSidebar"] button {
  background-color: #24221A !important;
  border: 1px solid #3A372B !important;
  border-radius: 6px !important;
  color: #D8D5C6 !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] button p,
[data-testid="stSidebar"] button span {
  color: #D8D5C6 !important;
}

/* Enabled Secondary Button Hover */
[data-testid="stSidebar"] button:hover:not(:disabled) {
  background-color: #38352A !important;
  border-color: #585342 !important;
}
[data-testid="stSidebar"] button:hover:not(:disabled) p,
[data-testid="stSidebar"] button:hover:not(:disabled) span {
  color: #FFFFFF !important;
}

/* Active / Primary Button */
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button[kind="primary"] {
  background-color: #A8442A !important;
  border: 1px solid #C05034 !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] span,
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[kind="primary"] span {
  color: #FFFFFF !important;
  font-weight: 600 !important;
}

/* Disabled Button in Sidebar */
[data-testid="stSidebar"] button:disabled {
  background-color: #181712 !important;
  border: 1px solid #28261E !important;
  opacity: 0.65 !important;
  cursor: not-allowed !important;
}
[data-testid="stSidebar"] button:disabled p,
[data-testid="stSidebar"] button:disabled span {
  color: #5C594A !important;
}

.oa-brand h1 {
  font-size: 16px;
  font-weight: 700;
  color: #F2F0E6;
  margin: 0;
  letter-spacing: -0.01em;
}
.oa-brand p {
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #75725F;
  margin-top: 4px;
}

/* Cards & Containers */
.oa-card {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 16px;
  margin-bottom: 12px;
}

.oa-filerow {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--rule);
  background: var(--surface);
  border-radius: 3px;
}
.oa-tick {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--full-bg);
  color: var(--full);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  flex: none;
}
.oa-tick.warn {
  background: var(--partial-bg);
  color: var(--partial);
}
.oa-nm {
  flex: 1;
  font-size: 13px;
  font-weight: 600;
}
.oa-meta {
  font-family: monospace;
  font-size: 11px;
  color: var(--ink3);
}
.oa-warnbox {
  font-size: 12.5px;
  color: var(--ink2);
  padding: 12px 14px;
  background: var(--partial-bg);
  border-radius: 4px;
  margin-top: 10px;
  border-left: 3px solid var(--partial);
}

/* Pills & Badges */
.oa-pill {
  display: inline-block;
  font-family: monospace;
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 2px;
  margin-left: 4px;
  text-transform: uppercase;
  font-weight: 600;
}
.oa-pill.ko {
  background: var(--accent-soft);
  color: var(--accent);
}
.oa-pill.std {
  background: #E2E6E9;
  color: #3E5666;
}
.oa-pill.custom {
  background: #EFEDE6;
  color: var(--ink2);
}

/* Verdict Badges */
.v-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 3px;
  font-family: monospace;
  font-size: 12px;
  font-weight: 700;
}
.v-full { background: var(--full-bg); color: var(--full); }
.v-partial { background: var(--partial-bg); color: var(--partial); }
.v-none { background: var(--none-bg); color: var(--none); }
.v-unclear { background: var(--unclear-bg); color: var(--unclear); }

/* Ranking Card */
.rcard {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 14px;
  height: 100%;
}
.rcard.win {
  border: 2px solid var(--ink);
  background: var(--white);
}
.rcard.dq {
  opacity: 0.65;
  background: #F4F2EB;
}
.rcard .pos {
  font-family: monospace;
  font-size: 10px;
  color: var(--ink3);
  font-weight: 600;
}
.rcard .nm {
  font-size: 14px;
  font-weight: 700;
  margin: 4px 0 6px;
}
.rcard .sc {
  font-family: monospace;
  font-size: 22px;
  font-weight: 700;
}
.rcard .sc s {
  font-size: 12px;
  color: var(--ink3);
  text-decoration: none;
  font-weight: 400;
}
.rcard .flag {
  font-family: monospace;
  font-size: 11px;
  color: var(--partial);
  margin-top: 6px;
}
.rcard .flag.bad {
  color: var(--none);
  font-weight: 700;
}
.rcard .flag.ok {
  color: var(--full);
}

/* Runbar */
.runbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 18px;
  background: var(--ink);
  color: #E4E1D2;
  border-radius: 4px;
  margin-bottom: 20px;
}

/* Legend */
.legend {
  display: flex;
  gap: 18px;
  align-items: center;
  margin: 14px 0;
  font-size: 12px;
  color: var(--ink2);
}
.lg-item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.lg-box {
  width: 20px;
  height: 20px;
  border-radius: 3px;
  display: grid;
  place-items: center;
  font-family: monospace;
  font-size: 11px;
  font-weight: 700;
}

/* Quote box */
.oa-quote {
  font-family: monospace;
  font-size: 12px;
  line-height: 1.6;
  padding: 12px 14px;
  background: var(--surface);
  border-left: 3px solid var(--ink);
  margin: 10px 0;
  color: var(--ink);
}

/* Memory Box */
.oa-memory {
  margin-top: 24px;
  padding: 16px 18px;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--rule2);
  border-radius: 4px;
}
.oa-memory h5 {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 4px;
}
.oa-memory p {
  font-size: 11.5px;
  color: var(--ink3);
  margin-bottom: 10px;
}
.oa-memory ul {
  padding-left: 18px;
  margin: 0;
}
.oa-memory li {
  font-size: 12.5px;
  color: var(--ink2);
  margin-bottom: 6px;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource
def get_conn():
    conn = get_db()
    init_db(conn)
    seed_if_empty(conn)
    return conn


conn = get_conn()


def load_rfp_from_db():
    row = conn.execute("SELECT * FROM rfp WHERE id = 1").fetchone()
    if not row:
        return None
    return {
        "filename": row["filename"],
        "pages": json.loads(row["pages_json"]),
        "is_scan": bool(row["is_scan"]),
        "size_kb": row["size_kb"],
    }


def save_rfp_to_db(rfp: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rfp (id, filename, pages_json, is_scan, size_kb) VALUES (1, ?, ?, ?, ?)",
        (rfp["filename"], json.dumps(rfp["pages"]), int(rfp["is_scan"]), rfp["size_kb"]),
    )
    conn.commit()


def clear_rfp_from_db() -> None:
    conn.execute("DELETE FROM rfp WHERE id = 1")
    conn.commit()


if "screen" not in st.session_state:
    st.session_state.screen = "upload"
if "rfp" not in st.session_state:
    # The RFP is persisted in the db (see rfp table) so it survives a browser
    # reload the same way uploaded offers already do — without this, a reload
    # reset session_state.rfp to None while offer_files() (db-backed) kept
    # showing the uploaded offers, an inconsistent "half reset" state.
    st.session_state.rfp = load_rfp_from_db()
if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # offer_id -> list of (user_msg, bot_msg, cites)


def offer_files():
    return conn.execute("SELECT * FROM offers ORDER BY id").fetchall()


def existing_offer_names():
    return {r["name"] for r in offer_files()}


def prepare_offer_vectors(show_status=True):
    offers = offer_files()
    if not offers:
        return []

    errors = []
    progress = None
    status = None

    if show_status:
        status = st.status("Preparing supplier offers for analysis...", expanded=True)
        status.write("Checking uploaded offers.")
        progress = st.progress(0, text="Starting analysis...")

    for idx, offer in enumerate(offers, start=1):
        label = f"{offer['supplier']} ({offer['name']})"
        try:
            pages = json.loads(offer["pages_json"])
            if show_status:
                progress.progress((idx - 1) / len(offers), text=f"Analyzing {label}")

            if not is_offer_indexed(offer["id"]):
                index_offer_chunks(offer["id"], pages)
        except Exception as err:
            errors.append(label)

    if show_status:
        progress.progress(1.0, text="Analysis preparation complete.")
        if errors:
            status.update(
                label="Some offers could not be prepared. Analysis will continue with the rest.",
                state="error",
                expanded=True,
            )
            for label in errors:
                status.write(f"Could not prepare {label}.")
        else:
            status.update(label="Offers are ready for analysis.", state="complete", expanded=False)

    return errors


def delete_offer_vectors(offer_ids):
    errors = []
    for offer_id in offer_ids:
        try:
            delete_offer_chunks(offer_id)
        except Exception as err:
            errors.append(str(err))
    return errors


def run_analysis_pipeline(rfp_text):
    with st.status("Analysing supplier offers...", expanded=True) as status:
        status.write("Preparing offer content for every uploaded offer.")
        vector_errors = prepare_offer_vectors(show_status=False)
        if vector_errors:
            status.write("Some offers had warnings during preparation. Analysis will continue.")

        status.write("Evaluating each offer against the RFP criteria.")
        run_grading_batch(conn, rfp_text)

        status.write("Saving supplier evaluation summaries.")
        try:
            store_tender_memories(conn)
        except Exception:
            status.write("Could not save evaluation summaries to memory.")

        status.update(label="Analysis complete.", state="complete", expanded=False)


def go_to_screen(screen, prepare=True):
    if prepare and screen in {"criteria", "grid", "chat"}:
        with st.spinner("Preparing uploaded offers for retrieval..."):
            prepare_offer_vectors(show_status=False)
    st.session_state.screen = screen
    st.rerun()


# ---------- Sidebar Navigation ----------
with st.sidebar:
    st.markdown(
        '<div class="oa-brand"><h1>RFP Offer Analyst</h1><p>Indirect procurement</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    
    n_offers = len(offer_files())
    rfp_ready = st.session_state.rfp is not None
    
    steps = [
        ("upload", "01  Upload", False, "Upload RFP & offer documents"),
        ("criteria", "02  Criteria", not (rfp_ready and n_offers > 0), "Upload RFP & at least 1 offer document to unlock Criteria"),
        ("grid", "03  Analysis grid", not (rfp_ready and n_offers > 0), "Upload RFP & at least 1 offer document to unlock Analysis Grid"),
        ("chat", "04  Chat with offer", not (rfp_ready and n_offers > 0), "Upload RFP & at least 1 offer document to unlock Chat"),
    ]
    
    for key, label, disabled, tooltip in steps:
        kind = "primary" if st.session_state.screen == key else "secondary"
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type=kind,
            disabled=disabled,
            help=tooltip if disabled else None,
        ):
            go_to_screen(key)
            
    if not (rfp_ready and n_offers > 0):
        missing = []
        if not rfp_ready:
            missing.append("RFP document")
        if n_offers == 0:
            missing.append("Offer document")
        st.markdown(
            f'<div style="font-size:11px;color:#A8705C;background:#261E1B;padding:8px 10px;border-radius:4px;margin-top:10px;line-height:1.4;">'
            f'🔒 <b>Steps 02–04 locked</b><br>Please upload: {", ".join(missing)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div style="font-size:11px;color:#75725F;line-height:1.5;padding-top:20px;border-top:1px solid #2E2C22">'
        'TND-2026-0418<br><span style="font-family:monospace">Fleet telematics platform</span></div>',
        unsafe_allow_html=True
    )


# ==============================================================================
# SCREEN 01: UPLOAD
# ==============================================================================
if st.session_state.screen == "upload":
    st.title("Documents")
    st.caption("One RFP, then every offer you received. Price sheets can ride along.")

    col_rfp, col_offers = st.columns(2)

    with col_rfp:
        st.markdown("**The ruler**")
        rfp_file = st.file_uploader(
            "RFP Document", type=["pdf", "docx", "xlsx"], key="rfp_uploader", label_visibility="collapsed"
        )
        if rfp_file is not None and (
            st.session_state.rfp is None or st.session_state.rfp["filename"] != rfp_file.name
        ):
            data = rfp_file.getvalue()
            pages, scan = extract(rfp_file.name, data)
            st.session_state.rfp = {
                "filename": rfp_file.name,
                "pages": pages,
                "is_scan": scan,
                "size_kb": round(len(data) / 1024, 1),
            }
            save_rfp_to_db(st.session_state.rfp)

        rfp = st.session_state.rfp
        if rfp:
            tick_cls = "warn" if rfp["is_scan"] else ""
            tick_glyph = "!" if rfp["is_scan"] else "✓"
            c_rfp1, c_rfp2 = st.columns([5, 1])
            with c_rfp1:
                st.markdown(
                    f'<div class="oa-filerow"><span class="oa-tick {tick_cls}">{tick_glyph}</span>'
                    f'<span class="oa-nm">{rfp["filename"]}</span>'
                    f'<span class="oa-meta">{len(rfp["pages"])} p · {rfp["size_kb"]} KB</span></div>',
                    unsafe_allow_html=True,
                )
            with c_rfp2:
                if st.button("🗑️", key="del_rfp", help="Remove RFP document"):
                    st.session_state.rfp = None
                    clear_rfp_from_db()
                    st.rerun()

            if rfp["is_scan"]:
                st.markdown(
                    f'<div class="oa-warnbox"><b>{rfp["filename"]}</b> yielded almost no text '
                    "— it looks like a scan. Grading against it now would mark every "
                    "criterion unclear. Run OCR first.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Everything is measured against this document.")

    with col_offers:
        rows = offer_files()
        st.markdown(f"**Offers received · {len(rows)}**")
        uploaded = st.file_uploader(
            "Supplier Offers",
            type=["pdf", "docx", "xlsx"],
            accept_multiple_files=True,
            key="offers_uploader",
            label_visibility="collapsed",
        )
        if uploaded:
            existing = existing_offer_names()
            for f in uploaded:
                if f.name in existing:
                    continue
                data = f.getvalue()
                pages, scan = extract(f.name, data)
                default_supplier = f.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
                conn.execute(
                    "INSERT INTO offers (name, supplier, pages_json, price) VALUES (?, ?, ?, ?)",
                    (f.name, default_supplier, json.dumps(pages), None),
                )
            conn.commit()
            st.rerun()

        rows = offer_files()
        scanned_names = []
        for r in rows:
            pages = json.loads(r["pages_json"])
            scan = is_scanned(pages)
            if scan:
                scanned_names.append(r["name"])
            tick_cls = "warn" if scan else ""
            tick_glyph = "!" if scan else "✓"

            c_hdr1, c_hdr2 = st.columns([5, 1])
            with c_hdr1:
                st.markdown(
                    f'<div class="oa-filerow"><span class="oa-tick {tick_cls}">{tick_glyph}</span>'
                    f'<span class="oa-nm">{r["name"]}</span>'
                    f'<span class="oa-meta">{len(pages)} p</span></div>',
                    unsafe_allow_html=True,
                )
            with c_hdr2:
                if st.button("🗑️", key=f"del_offer_{r['id']}", help=f"Delete {r['name']}"):
                    delete_offer_vectors([r["id"]])
                    conn.execute("DELETE FROM evaluations WHERE offer_id = ?", (r["id"],))
                    conn.execute("DELETE FROM offers WHERE id = ?", (r["id"],))
                    conn.commit()
                    st.rerun()

            c1, c2 = st.columns(2)
            new_supplier = c1.text_input(
                "Supplier", value=r["supplier"], key=f"supplier_{r['id']}", label_visibility="collapsed"
            )
            new_price = c2.text_input(
                "Price", value=r["price"] or "", key=f"price_{r['id']}", label_visibility="collapsed",
                placeholder="Price (optional)",
            )
            if new_supplier != r["supplier"] or new_price != (r["price"] or ""):
                conn.execute(
                    "UPDATE offers SET supplier = ?, price = ? WHERE id = ?",
                    (new_supplier, new_price or None, r["id"]),
                )
                conn.commit()

        if scanned_names:
            names = ", ".join(f"<b>{n}</b>" for n in scanned_names)
            st.markdown(
                f'<div class="oa-warnbox">{names} yielded almost no text — '
                "it looks like a scan. Grading it now would mark every criterion unclear. "
                "Run OCR first.</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    action_col1, action_col2 = st.columns([2, 2])
    with action_col1:
        if st.button("Continue to criteria →", type="primary", disabled=not (st.session_state.rfp and offer_files())):
            prepare_offer_vectors(show_status=True)
            go_to_screen("criteria", prepare=False)

    with action_col2:
        if (st.session_state.rfp or offer_files()) and st.button("Clear all documents", key="clear_all"):
            delete_offer_vectors([r["id"] for r in offer_files()])
            st.session_state.rfp = None
            clear_rfp_from_db()
            conn.execute("DELETE FROM evaluations")
            conn.execute("DELETE FROM offers")
            conn.commit()
            st.rerun()

    if not st.session_state.rfp or not offer_files():
        st.caption("Upload the RFP and at least one offer to continue.")


# ==============================================================================
# SCREEN 02: CRITERIA
# ==============================================================================
elif st.session_state.screen == "criteria":
    st.title("Criteria")
    n_standard = conn.execute("SELECT COUNT(*) FROM criteria WHERE source='standard'").fetchone()[0]
    st.caption(f"{n_standard} standard checks, plus anything specific to this tender.")

    crit_rows = conn.execute("SELECT * FROM criteria ORDER BY id").fetchall()

    header = st.columns([5, 2, 1.5, 2])
    header[0].markdown("**Requirement**")
    header[1].markdown("**Weight**")
    header[2].markdown("**Knockout**")
    header[3].markdown("**Source**")

    for c in crit_rows:
        row = st.columns([5, 2, 1.5, 2])
        row[0].write(c["question"])

        wcols = row[1].columns([1, 1, 1])
        if wcols[0].button("−", key=f"wminus_{c['id']}"):
            new_w = max(1, c["weight"] - 1)
            conn.execute("UPDATE criteria SET weight = ? WHERE id = ?", (new_w, c["id"]))
            conn.commit()
            st.rerun()
        wcols[1].markdown(f"<div style='text-align:center;padding-top:4px;font-weight:600'>{c['weight']}</div>", unsafe_allow_html=True)
        if wcols[2].button("+", key=f"wplus_{c['id']}"):
            new_w = min(9, c["weight"] + 1)
            conn.execute("UPDATE criteria SET weight = ? WHERE id = ?", (new_w, c["id"]))
            conn.commit()
            st.rerun()

        new_ko = row[2].toggle("KO", value=bool(c["is_knockout"]), key=f"ko_{c['id']}", label_visibility="collapsed")
        if new_ko != bool(c["is_knockout"]):
            conn.execute("UPDATE criteria SET is_knockout = ? WHERE id = ?", (int(new_ko), c["id"]))
            conn.commit()
            st.rerun()

        pill = '<span class="oa-pill std">standard</span>' if c["source"] == "standard" else '<span class="oa-pill custom">user</span>'
        if c["is_knockout"]:
            pill += '<span class="oa-pill ko">KO</span>'
        row[3].markdown(pill, unsafe_allow_html=True)

    st.write("")
    with st.form("add_criterion", clear_on_submit=True):
        new_q = st.text_input(
            "Add a requirement",
            placeholder="Add a requirement in plain language — e.g. must provide German-speaking on-site support",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Add criterion")
        if submitted and new_q.strip():
            conn.execute(
                "INSERT INTO criteria (question, weight, is_knockout, source) VALUES (?, 3, 0, 'user')",
                (new_q.strip(),),
            )
            conn.commit()
            st.rerun()

    n_offers = len(offer_files())
    st.caption(f"Adding one criterion queues {n_offers} new checks — one per offer. Nothing else is re-graded.")
    
    st.write("")
    if st.button("Run analysis →", type="primary"):
        st.session_state.screen = "grid"
        st.rerun()


# ==============================================================================
# SCREEN 03: ANALYSIS GRID
# ==============================================================================
elif st.session_state.screen == "grid":
    top_col1, top_col2 = st.columns([4, 1])
    top_col1.title("Analysis grid")
    top_col1.caption("Every offer graded alone against the same ruler. Click any cell for the evidence.")
    
    if top_col2.button("Re-run analysis", use_container_width=True):
        rfp_text = ""
        if st.session_state.rfp and st.session_state.rfp.get("pages"):
            rfp_text = "\n\n".join(txt for _, txt in st.session_state.rfp["pages"])
        run_analysis_pipeline(rfp_text)
        st.rerun()

    # Check if evaluations exist, if not run automatically
    n_evals = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    n_criteria = conn.execute("SELECT COUNT(*) FROM criteria").fetchone()[0]
    n_offers = len(offer_files())
    total_checks = n_criteria * n_offers

    if n_evals < total_checks:
        rfp_text = ""
        if st.session_state.rfp and st.session_state.rfp.get("pages"):
            rfp_text = "\n\n".join(txt for _, txt in st.session_state.rfp["pages"])
        run_analysis_pipeline(rfp_text)
        st.rerun()

    # Calculate rankings & scores
    ranked_offers, max_score = calculate_scores(conn)

    # Status Bar
    st.markdown(
        f'<div class="runbar">'
        f'<div><b>{n_evals} / {total_checks}</b> checks evaluated</div>'
        f'<div style="flex:1;height:4px;background:#33301F;border-radius:2px;overflow:hidden">'
        f'<div style="height:100%;width:{min(100, int(n_evals/total_checks*100))}%;background:var(--accent)"></div></div>'
        f'<div style="font-family:monospace;font-size:12px">complete · 20 concurrent</div></div>',
        unsafe_allow_html=True,
    )

    # Ranking Cards
    st.markdown("**Ranking**")
    st.caption("Score in points, not checks — full match = 2× a requirement's weight, partial = 1×, summed across all requirements.")
    rank_cols = st.columns(len(ranked_offers))
    for idx, r in enumerate(ranked_offers):
        is_win = (idx == 0 and not r["is_disqualified"])
        card_cls = "rcard win" if is_win else ("rcard dq" if r["is_disqualified"] else "rcard")
        pos_str = "DISQUALIFIED" if r["is_disqualified"] else f"#{idx + 1}"
        score_str = "—" if r["is_disqualified"] else str(r["score"])
        max_str = "" if r["is_disqualified"] else f" / {max_score} pts"
        
        flag_cls = "flag bad" if r["is_disqualified"] else ("flag" if r["unclear_count"] > 0 else "flag ok")
        flag_str = r["dq_reason"] if r["is_disqualified"] else (f"{r['unclear_count']} unclear — ask them" if r["unclear_count"] > 0 else "complete")
        
        ratio = 0 if r["is_disqualified"] or max_score == 0 else int(r["score"] / max_score * 100)

        with rank_cols[idx]:
            st.markdown(
                f'<div class="{card_cls}">'
                f'<div class="pos">{pos_str}</div>'
                f'<div class="nm">{r["supplier"]}</div>'
                f'<div class="sc">{score_str}<s>{max_str}</s></div>'
                f'<div style="height:3px;background:var(--rule);margin:8px 0">'
                f'<div style="height:100%;width:{ratio}%;background:var(--ink)"></div></div>'
                f'<div class="{flag_cls}">{flag_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Legend
    st.markdown(
        '<div class="legend">'
        '<div class="lg-item"><div class="lg-box v-full">🟢 F</div> Full — meets the requirement</div>'
        '<div class="lg-item"><div class="lg-box v-partial">🟡 P</div> Partial — meets it in part</div>'
        '<div class="lg-item"><div class="lg-box v-none">🔴 N</div> None — fails the requirement</div>'
        '<div class="lg-item"><div class="lg-box v-unclear">⚪ ?</div> Silent — offer doesn\'t mention it</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Hover any cell for the reason and confidence · click to see the full evidence.")

    # Matrix Grid Table
    all_criteria = conn.execute("SELECT * FROM criteria ORDER BY id").fetchall()
    offers_list = offer_files()

    # Build Grid Header
    header_cols = st.columns([4] + [2] * len(offers_list))
    header_cols[0].markdown("**Requirement**")
    for o_idx, off in enumerate(offers_list):
        p_str = f"<br><span style='font-family:monospace;font-size:11px;color:var(--ink3)'>{off['price']}</span>" if off['price'] else ""
        header_cols[o_idx + 1].markdown(
            f"**{off['supplier']}**{p_str}", unsafe_allow_html=True
        )

    # Build Rows
    GLYPH = {"full": "🟢 F", "partial": "🟡 P", "none": "🔴 N", "unclear": "⚪ ?"}
    LABEL = {
        "full": "Full conformance — meets the requirement",
        "partial": "Partial conformance — meets it in part",
        "none": "No conformance — fails the requirement",
        "unclear": "Offer is silent — not mentioned",
    }

    for crit in all_criteria:
        r_cols = st.columns([4] + [2] * len(offers_list))
        ko_badge = '<span class="oa-pill ko">KO</span>' if crit["is_knockout"] else ""
        r_cols[0].markdown(
            f"{crit['question']} {ko_badge} <span style='font-family:monospace;font-size:10.5px;color:var(--ink3)'>×{crit['weight']}</span>",
            unsafe_allow_html=True,
        )

        for o_idx, off in enumerate(offers_list):
            ev = conn.execute(
                "SELECT * FROM evaluations WHERE offer_id = ? AND criterion_id = ?",
                (off["id"], crit["id"]),
            ).fetchone()

            verdict = ev["verdict"] if ev else "unclear"
            glyph = GLYPH.get(verdict, "⚪ ?")
            btn_key = f"cell_{crit['id']}_{off['id']}"

            tooltip = LABEL.get(verdict, "Not yet evaluated")
            if ev and ev["reason"]:
                tooltip += f"\n\nWhy: {ev['reason']}"
            if ev and ev["confidence"]:
                tooltip += f"\nConfidence: {ev['confidence']}"

            if r_cols[o_idx + 1].button(
                glyph, key=btn_key, use_container_width=True, help=tooltip
            ):
                st.session_state.selected_cell = (crit["id"], off["id"])

    # Cell Inspector Modal / Dialog
    if st.session_state.selected_cell:
        c_id, o_id = st.session_state.selected_cell
        crit_item = conn.execute("SELECT * FROM criteria WHERE id = ?", (c_id,)).fetchone()
        offer_item = conn.execute("SELECT * FROM offers WHERE id = ?", (o_id,)).fetchone()
        ev_item = conn.execute(
            "SELECT * FROM evaluations WHERE offer_id = ? AND criterion_id = ?",
            (o_id, c_id),
        ).fetchone()

        if crit_item and offer_item and ev_item:
            @st.dialog("Evaluation Evidence")
            def show_inspector():
                v = ev_item["verdict"]
                lbl = {"full": "Full conformance", "partial": "Partial conformance", "none": "No conformance", "unclear": "Offer is silent"}.get(v, v)
                glyph = GLYPH.get(v, "?")
                
                st.caption(f"{offer_item['supplier']} · {offer_item['name']}")
                st.markdown(f"### {crit_item['question']}")
                
                st.markdown(
                    f'<span class="v-badge v-{v}">{glyph}  {lbl}</span>',
                    unsafe_allow_html=True,
                )
                st.write("")
                st.markdown(f"**Reason:** {ev_item['reason']}")
                
                quote_text = ev_item['quote']
                if quote_text:
                    st.markdown(f'<div class="oa-quote">“{quote_text}”</div>', unsafe_allow_html=True)
                else:
                    st.caption("No supporting quote — the offer does not explicitly address this requirement.")
                
                st.divider()
                st.markdown(f"- **Page number:** p. {ev_item['page']}")
                st.markdown(f"- **Confidence:** {ev_item['confidence']}")
                st.markdown(f"- **Weight:** ×{crit_item['weight']} {'(Knockout)' if crit_item['is_knockout'] else ''}")
                
                st.write("")
                st.markdown("**Override Verdict:**")
                oc1, oc2, oc3, oc4 = st.columns(4)
                for new_v in ["full", "partial", "none", "unclear"]:
                    if oc1.button(new_v.capitalize(), key=f"ov_{new_v}"):
                        conn.execute(
                            "UPDATE evaluations SET verdict = ? WHERE offer_id = ? AND criterion_id = ?",
                            (new_v, o_id, c_id)
                        )
                        conn.commit()
                        st.session_state.selected_cell = None
                        st.rerun()

            show_inspector()

    # Memory Box
    st.write("")
    notes = get_tender_memory_notes(conn)
    notes_html = "".join(f"<li>{n}</li>" for n in notes)
    st.markdown(
        f'<div class="oa-memory">'
        f'<h5>From past tenders</h5>'
        f'<p>Supplier history recalled from memory. Background context for buyer — does not alter verdicts above.</p>'
        f'<ul>{notes_html}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# SCREEN 04: CHAT WITH OFFER
# ==============================================================================
elif st.session_state.screen == "chat":
    st.title("Chat with an offer")
    st.caption("For the questions the checklist didn't cover. One supplier at a time.")

    offers_list = offer_files()
    if not offers_list:
        st.info("Upload at least one offer document to start chat.")
    else:
        chat_col_left, chat_col_right = st.columns([1, 3])

        with chat_col_left:
            st.markdown("**Selected Offer**")
            offer_opts = {o["id"]: f"{o['supplier']} ({o['name']})" for o in offers_list}
            selected_o_id = st.radio(
                "Select Offer",
                options=list(offer_opts.keys()),
                format_func=lambda x: offer_opts[x],
                label_visibility="collapsed",
            )
            st.caption("Retrieval is filtered to the selected supplier. Documents never mix across suppliers.")

        selected_offer = next(o for o in offers_list if o["id"] == selected_o_id)

        with chat_col_right:
            st.markdown(f"**Thread with {selected_offer['supplier']}**")
            
            if selected_o_id not in st.session_state.chat_history:
                st.session_state.chat_history[selected_o_id] = []

            # Display Suggestion Chips
            suggestions = [
                "What is excluded from scope?",
                "Any hidden fees or surcharges?",
                "Can we terminate early?",
                "Who are the subcontractors?",
            ]
            
            chip_cols = st.columns(len(suggestions))
            for i, sug in enumerate(suggestions):
                if chip_cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
                    with st.spinner("drafting an answer..."):
                        res = answer_offer_question(
                            offer_id=selected_o_id,
                            supplier_name=selected_offer["supplier"],
                            query=sug,
                            pages_json=selected_offer["pages_json"],
                        )
                    st.session_state.chat_history[selected_o_id].append((sug, res["answer"], res["citations"]))
                    st.rerun()

            st.write("")
            
            # Chat Container
            chat_container = st.container()
            with chat_container:
                history = st.session_state.chat_history[selected_o_id]
                if not history:
                    st.caption(f"Ask anything about {selected_offer['supplier']}'s proposal.")
                else:
                    for user_msg, bot_msg, cites in history:
                        with st.chat_message("user"):
                            st.write(user_msg)
                        with st.chat_message("assistant"):
                            st.write(bot_msg)

            # Chat Input Form
            prompt = st.chat_input(f"Ask about {selected_offer['supplier']}'s offer...")
            if prompt:
                with st.spinner("Retrieving matching offer details and drafting an answer..."):
                    res = answer_offer_question(
                        offer_id=selected_o_id,
                        supplier_name=selected_offer["supplier"],
                        query=prompt,
                        pages_json=selected_offer["pages_json"],
                    )
                st.session_state.chat_history[selected_o_id].append((prompt, res["answer"], res["citations"]))
                st.rerun()
