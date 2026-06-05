"""
PDF Cross-Validator — Streamlit UI
Run with:  python -m streamlit run app.py
"""
import streamlit as st
import tempfile, os, re
from io import StringIO
import contextlib
from dataclasses import asdict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

_ANSI = re.compile(r'\033\[[0-9;]*m')
def _strip(s): return _ANSI.sub('', s)

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Cross-Validator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"]  { display: none; }
    </style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────
st.title("📋 PDF Cross-Validator")
st.caption(
    "Validate component sub-PDFs against a main GAD / BOM drawing · "
    "regex extraction + Gemini AI"
)

if not GEMINI_KEY or GEMINI_KEY == "your_key_here":
    st.warning(
        "No Gemini API key found. Add `GEMINI_API_KEY=...` to the `.env` file "
        "for AI-powered field recovery and final judge.",
        icon="⚠️",
    )

st.divider()

# ── upload ────────────────────────────────────────────────────
st.header("1️⃣  Upload PDFs")
col_l, col_r = st.columns([1, 2])

with col_l:
    st.subheader("Main PDF (GAD / BOM)")
    main_file = st.file_uploader("main", type="pdf", label_visibility="collapsed")
    if main_file:
        st.success(f"✔  {main_file.name}", icon="📄")

with col_r:
    st.subheader("Sub-PDFs (component drawings)")
    sub_files = st.file_uploader(
        "subs", type="pdf", accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if sub_files:
        for f in sub_files:
            st.write(f"• {f.name}")

st.divider()

# ── options row ───────────────────────────────────────────────
opt_col, _, run_col = st.columns([1, 2, 1])
with opt_col:
    debug_mode = st.checkbox("Debug mode", help="Show all extracted KV fields per sub-PDF")
with run_col:
    ready = main_file is not None and bool(sub_files)
    run_clicked = st.button(
        "▶  Run Validation", type="primary",
        disabled=not ready, use_container_width=True,
    )

if not ready:
    st.info("⬆️  Upload a main PDF and at least one sub-PDF to begin.")
    st.stop()

if not run_clicked:
    st.stop()

# ── execute ───────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    main_path = os.path.join(tmp, main_file.name)
    with open(main_path, "wb") as fh:
        fh.write(main_file.getvalue())

    sub_paths = []
    for sf in sub_files:
        p = os.path.join(tmp, sf.name)
        with open(p, "wb") as fh:
            fh.write(sf.getvalue())
        sub_paths.append(p)

    json_path = os.path.join(tmp, "report.json")

    log_buf = StringIO()
    error   = None

    with st.spinner("Validating — this may take a moment…"):
        try:
            with contextlib.redirect_stdout(log_buf):
                from test import run as _run
                result = _run(
                    main_path, sub_paths,
                    output=json_path,
                    gemini_key=GEMINI_KEY,
                    debug=debug_mode,
                )
            if isinstance(result, tuple) and len(result) == 3:
                main_spec, main_bom, reports = result
            else:
                main_spec, main_bom, reports = {}, [], result

            with open(json_path, "rb") as fh:
                json_bytes = fh.read()

        except Exception as exc:
            error = exc

if error:
    st.error(f"Validation error: {error}")
    with st.expander("Log output"):
        st.code(_strip(log_buf.getvalue()))
    st.stop()

# ── results ───────────────────────────────────────────────────
st.header("2️⃣  Results")

total  = len(reports)
passed = sum(1 for r in reports if r.overall_pass)
gcalls = sum(1 for r in reports if r.gemini_used)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sub-PDFs checked", total)
m2.metric(
    "Fully passed", f"{passed} / {total}",
    delta="all clear" if passed == total else f"{total - passed} with failures",
    delta_color="off" if passed == total else "inverse",
)
m3.metric("Gemini calls used", gcalls)
m4.download_button(
    "⬇  Download JSON report",
    data=json_bytes,
    file_name="validation_report.json",
    mime="application/json",
    use_container_width=True,
)

st.divider()

# Main spec
if main_spec:
    with st.expander(f"📄 Main PDF — Spec ({len(main_spec)} fields extracted)"):
        spec_df = pd.DataFrame(main_spec.items(), columns=["Field", "Value"])
        st.dataframe(spec_df, use_container_width=True, hide_index=True)

# BOM
if main_bom:
    with st.expander(f"📋 Main PDF — BOM ({len(main_bom)} items)"):
        bom_df = pd.DataFrame([asdict(b) for b in main_bom])
        st.dataframe(bom_df, use_container_width=True, hide_index=True)

# ── per sub-PDF reports ───────────────────────────────────────
st.subheader("Sub-PDF Reports")

for rep in reports:
    icon   = "✅" if rep.overall_pass else "❌"
    ai_tag = "  🤖 Gemini" if rep.gemini_used else ""
    label  = f"{icon}  {rep.sub_pdf}{ai_tag}"

    with st.expander(label, expanded=True):
        if not rep.matched_bom_item:
            st.warning("No BOM row matched — validation skipped.")
            continue

        b = rep.matched_bom_item
        st.caption(f"Matched BOM row **[{b.sr_no}]** — {b.description}")

        if not rep.results:
            st.info("No fields to validate.")
            continue

        rows = [{
            "Status":          "✔ PASS" if r.match else "✗ FAIL",
            "Field":           r.field,
            "Expected (BOM)":  r.main_value,
            "Found (Drawing)": r.sub_value,
            "Similarity":      f"{r.similarity:.0f}%",
            "AI":              "🤖" if r.gemini_resolved else "",
        } for r in rep.results]

        df = pd.DataFrame(rows)

        def _row_color(row):
            bg = "#1a4731" if row["Status"] == "✔ PASS" else "#5c1515"
            return [f"background-color:{bg}; color:#f0f0f0"] * len(row)

        st.dataframe(
            df.style.apply(_row_color, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        if rep.overall_pass:
            st.success("All fields validated successfully.")
        else:
            failed = [r.field for r in rep.results if not r.match]
            st.error(f"Failed fields: {', '.join(failed)}")

        if debug_mode and rep.sub_kv:
            with st.expander("Extracted KV fields (debug)"):
                kv_df = pd.DataFrame(rep.sub_kv.items(), columns=["Key", "Value"])
                st.dataframe(kv_df, use_container_width=True, hide_index=True)

# Raw log
with st.expander("🔍 Raw log output"):
    st.code(_strip(log_buf.getvalue()), language=None)
