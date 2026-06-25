"""
PDF Cross-Validator — Streamlit UI
Run with:  python -m streamlit run app.py
"""
import streamlit as st
import tempfile, os, re, io
from io import StringIO
import contextlib
from dataclasses import asdict
import pandas as pd
from dotenv import load_dotenv
from test import render_highlight

load_dotenv()
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
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

if not ready and "result" not in st.session_state:
    st.info("⬆️  Upload a main PDF and at least one sub-PDF to begin.")
    st.stop()

# ── execute ───────────────────────────────────────────────────
if run_clicked:
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
        log_buf   = StringIO()

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

                st.session_state["result"] = {
                    "main_spec":      main_spec,
                    "main_bom":       main_bom,
                    "reports":        reports,
                    "json_bytes":     json_bytes,
                    "log":            log_buf.getvalue(),
                    "main_pdf_bytes": main_file.getvalue(),
                    "error":          None,
                }
            except Exception as exc:
                st.session_state["result"] = {
                    "error": str(exc),
                    "log":   log_buf.getvalue(),
                }

if "result" not in st.session_state:
    st.stop()

res = st.session_state["result"]

if res.get("error"):
    st.error(f"Validation error: {res['error']}")
    with st.expander("Log output"):
        st.code(_strip(res.get("log", "")))
    st.stop()

main_spec      = res["main_spec"]
main_bom       = res["main_bom"]
reports        = res["reports"]
json_bytes     = res["json_bytes"]
main_pdf_bytes = res["main_pdf_bytes"]

# ── Dialog + iframe width fix ─────────────────────────────────
# 1. Widen the Streamlit dialog to ~92 vw
# 2. Force every iframe inside the dialog (st.components.v1.html)
#    to fill 100% of the dialog width instead of the default ~300 px
st.markdown("""
<style>
  div[data-testid="stDialog"] > div > div[role="dialog"] {
      width: min(1200px, 92vw) !important;
      max-width: 92vw !important;
  }
  /* Force the Streamlit component container and its iframe to fill the dialog */
  div[data-testid="stDialog"] [data-testid="stCustomComponentV1"],
  div[data-testid="stDialog"] [data-testid="stCustomComponentV1"] > div {
      width: 100% !important;
      min-width: 0 !important;
      max-width: 100% !important;
      display: block !important;
  }
  div[data-testid="stDialog"] iframe {
      width: 100% !important;
      min-width: 100% !important;
      max-width: 100% !important;
      display: block !important;
  }
</style>
""", unsafe_allow_html=True)

# Viewer height in pixels
_VIEWER_H = 680

# ── "View in GAD" popup ───────────────────────────────────────
@st.dialog("View in GAD", width="large")
def _view_in_gad_dialog(ref, label):
    import base64

    st.markdown(
        f"<p style='margin:0 0 10px;color:#999;font-size:13px;letter-spacing:.3px'>"
        f"GAD page {ref['page']} &nbsp;·&nbsp; "
        f"<b style='color:#e0e0e0;font-size:14px'>{label}</b>"
        f"&nbsp;&nbsp;<span style='color:#555'>— scroll to zoom &nbsp;·&nbsp; drag to pan</span>"
        f"</p>",
        unsafe_allow_html=True,
    )

    png = render_highlight(io.BytesIO(main_pdf_bytes), ref, full_page=True)
    b64 = base64.b64encode(png).decode()

    st.components.v1.html(
        f"""
        <style>
          /* Reset so the iframe body fills its full allocated width */
          *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
          html, body {{
            width: 100%;
            height: 100%;
            background: transparent;
            overflow: hidden;
          }}

          #viewer {{
            width: 100%;
            height: {_VIEWER_H}px;
            background: #111;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            overflow: hidden;
            cursor: grab;
            position: relative;
            user-select: none;
            box-shadow: 0 4px 24px rgba(0,0,0,.5);
            box-sizing: border-box;
          }}
          #viewer.dragging {{ cursor: grabbing; }}

          #img-wrap {{
            position: absolute;
            top: 0; left: 0;
            transform-origin: 0 0;
            will-change: transform;
          }}
          #img-wrap img {{ display: block; max-width: none; }}

          /* top-right pill: zoom % */
          #zoom-badge {{
            position: absolute;
            top: 12px; right: 12px;
            background: rgba(0,0,0,.55);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255,255,255,.1);
            color: #ccc;
            font-size: 12px;
            font-family: monospace;
            border-radius: 20px;
            padding: 4px 10px;
            pointer-events: none;
            z-index: 20;
          }}

          /* bottom-center controls */
          #controls {{
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 20;
            background: rgba(20,20,20,.7);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 30px;
            padding: 6px 14px;
          }}
          .ctrl-btn {{
            background: rgba(255,255,255,.07);
            border: 1px solid rgba(255,255,255,.12);
            color: #ddd;
            border-radius: 8px;
            width: 34px; height: 34px;
            font-size: 18px;
            line-height: 1;
            cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            transition: background .15s;
          }}
          .ctrl-btn:hover {{ background: rgba(255,255,255,.15); }}
          .ctrl-sep {{
            width: 1px; height: 22px;
            background: rgba(255,255,255,.12);
            margin: 0 2px;
          }}
        </style>

        <div id="viewer">
          <div id="img-wrap">
            <img id="gad-img" src="data:image/png;base64,{b64}" draggable="false"/>
          </div>
          <div id="zoom-badge">100%</div>
          <div id="controls">
            <button class="ctrl-btn" onclick="zoomBy(-0.2)" title="Zoom out">−</button>
            <button class="ctrl-btn" onclick="zoomBy(0.2)"  title="Zoom in">+</button>
            <div class="ctrl-sep"></div>
            <button class="ctrl-btn" onclick="resetView()"  title="Fit to window" style="font-size:14px">⊡</button>
          </div>
        </div>

        <script>
          const viewer = document.getElementById('viewer');
          const wrap   = document.getElementById('img-wrap');
          const img    = document.getElementById('gad-img');
          const badge  = document.getElementById('zoom-badge');

          let scale = 1, tx = 0, ty = 0;
          let dragging = false, startX, startY, startTx, startTy;

          function apply() {{
            wrap.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;
            badge.textContent    = Math.round(scale * 100) + '%';
          }}

          function clamp() {{
            const vw = viewer.clientWidth, vh = viewer.clientHeight;
            const iw = img.naturalWidth * scale, ih = img.naturalHeight * scale;
            const margin = 60;
            tx = Math.max(margin - iw, Math.min(vw - margin, tx));
            ty = Math.max(margin - ih, Math.min(vh - margin, ty));
          }}

          function fitToView() {{
            if (!img.naturalWidth) return;
            const vw = viewer.clientWidth, vh = viewer.clientHeight;
            scale = Math.min(vw / img.naturalWidth, vh / img.naturalHeight) * 0.95;
            tx = (vw - img.naturalWidth  * scale) / 2;
            ty = (vh - img.naturalHeight * scale) / 2;
            apply();
          }}

          function zoomBy(delta, cx, cy) {{
            const vw = viewer.clientWidth, vh = viewer.clientHeight;
            cx = (cx !== undefined) ? cx : vw / 2;
            cy = (cy !== undefined) ? cy : vh / 2;
            const prev = scale;
            scale = Math.min(10, Math.max(0.08, scale * (1 + delta)));
            const r = scale / prev;
            tx = cx - r * (cx - tx);
            ty = cy - r * (cy - ty);
            clamp(); apply();
          }}

          function resetView() {{ fitToView(); }}

          // ── mouse wheel zoom ──────────────────────────────────
          viewer.addEventListener('wheel', e => {{
            e.preventDefault();
            const rect = viewer.getBoundingClientRect();
            zoomBy(e.deltaY < 0 ? 0.12 : -0.12,
                   e.clientX - rect.left, e.clientY - rect.top);
          }}, {{ passive: false }});

          // ── drag to pan ───────────────────────────────────────
          viewer.addEventListener('mousedown', e => {{
            dragging = true; viewer.classList.add('dragging');
            startX = e.clientX; startY = e.clientY;
            startTx = tx; startTy = ty;
          }});
          window.addEventListener('mousemove', e => {{
            if (!dragging) return;
            tx = startTx + e.clientX - startX;
            ty = startTy + e.clientY - startY;
            clamp(); apply();
          }});
          window.addEventListener('mouseup', () => {{
            dragging = false; viewer.classList.remove('dragging');
          }});

          // ── pinch-to-zoom (touch) ─────────────────────────────
          let t0dist = null, t0scale, tMidX, tMidY;
          viewer.addEventListener('touchstart', e => {{
            if (e.touches.length === 2) {{
              const [a, b] = e.touches;
              t0dist  = Math.hypot(b.clientX-a.clientX, b.clientY-a.clientY);
              t0scale = scale;
              const rect = viewer.getBoundingClientRect();
              tMidX = (a.clientX+b.clientX)/2 - rect.left;
              tMidY = (a.clientY+b.clientY)/2 - rect.top;
            }} else if (e.touches.length === 1) {{
              dragging = true;
              startX = e.touches[0].clientX; startY = e.touches[0].clientY;
              startTx = tx; startTy = ty;
            }}
          }}, {{ passive: true }});
          viewer.addEventListener('touchmove', e => {{
            e.preventDefault();
            if (e.touches.length === 2 && t0dist) {{
              const [a, b] = e.touches;
              const d = Math.hypot(b.clientX-a.clientX, b.clientY-a.clientY);
              const prev = scale;
              scale = Math.min(10, Math.max(0.08, t0scale * d / t0dist));
              const r = scale / prev;
              tx = tMidX - r*(tMidX-tx); ty = tMidY - r*(tMidY-ty);
              clamp(); apply();
            }} else if (e.touches.length === 1 && dragging) {{
              tx = startTx + e.touches[0].clientX - startX;
              ty = startTy + e.touches[0].clientY - startY;
              clamp(); apply();
            }}
          }}, {{ passive: false }});
          viewer.addEventListener('touchend', () => {{ dragging=false; t0dist=null; }});

          // ── measure dialog width via parent viewport ─────────
          // window.parent is accessible (srcdoc + allow-same-origin).
          // Streamlit dialog CSS: width = min(1200px, 92vw), padding ~24px each side.
          function getDialogWidth() {{
            try {{
              const vw = window.parent.innerWidth;
              if (vw > 0) return Math.min(1200, vw * 0.92) - 48;
            }} catch(e) {{}}
            // Fallback: whatever width the iframe itself was allocated
            return document.documentElement.clientWidth;
          }}

          function syncSize() {{
            const w = getDialogWidth();
            viewer.style.width = w + 'px';

            // The iframe itself is narrow — stretch it and its wrapper divs
            // window.frameElement is accessible because sandbox has allow-same-origin
            try {{
              let el = window.frameElement;
              // Walk up ~5 levels: iframe → stCustomComponentV1 wrappers
              for (let i = 0; i < 5 && el && el.tagName !== 'BODY'; i++) {{
                el.style.setProperty('width',     w + 'px', 'important');
                el.style.setProperty('min-width', w + 'px', 'important');
                el.style.setProperty('max-width', w + 'px', 'important');
                el = el.parentElement;
              }}
            }} catch(e) {{}}
          }}

          // Retry until the viewer is actually wide (dialog may not be laid out yet)
          let _retries = 0;
          function syncAndRetry() {{
            syncSize();
            fitToView();
            if (viewer.offsetWidth < 400 && _retries < 15) {{
              _retries++;
              setTimeout(syncAndRetry, 80);
            }}
          }}

          window.addEventListener('resize', () => {{ syncSize(); fitToView(); }});

          // ── init ──────────────────────────────────────────────
          img.complete && img.naturalWidth
            ? syncAndRetry()
            : img.addEventListener('load', syncAndRetry);
        </script>
        """,
        height=_VIEWER_H + 4,
        scrolling=False,
    )

# ── results ───────────────────────────────────────────────────
st.header("2️⃣  Results")

total  = len(reports)
passed = sum(1 for r in reports if r.overall_pass)

m1, m2, m3 = st.columns(3)
m1.metric("Sub-PDFs checked", total)
m2.metric(
    "Fully passed", f"{passed} / {total}",
    delta="all clear" if passed == total else f"{total - passed} with failures",
    delta_color="off" if passed == total else "inverse",
)
m3.download_button(
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
    icon  = "✅" if rep.overall_pass else "❌"
    label = f"{icon}  {rep.sub_pdf}"

    with st.expander(label, expanded=True):
        if not rep.matched_bom_item:
            st.warning("No BOM row matched — validation skipped.")
            continue

        b = rep.matched_bom_item
        st.caption(f"Matched BOM row **[{b.sr_no}]** — {b.description}")

        table_rows = [{
            "status": "✔ PASS" if r.match else "✗ FAIL",
            "field": r.field, "expected": r.main_value, "found": r.sub_value,
            "similarity": f"{r.similarity:.0f}%", "gad_ref": r.gad_ref,
        } for r in rep.results]
        table_rows += [{
          "status": "-", "field": "-", "expected": o.value, "found": o.value,
          "similarity": "100%", "gad_ref": o.gad_ref,
        } for o in rep.orphan_values]

        if not table_rows:
            st.info("No fields to validate.")
            continue

        COL_W = [1, 1.3, 2, 2, 1]
        # PDF Reference column hidden from UI; keep the row data and View code
        # below commented so the column can be restored later if needed.
        # COL_W = [1, 1.3, 2, 2, 1, 1.2]
        hdr_cols = st.columns(COL_W)
        for col, label in zip(hdr_cols, ["Status", "Field", "Main Drawing ",
                  "Sub Drawing ", "Similarity"]):
                  # "PDF Reference"]):
            col.markdown(f"**{label}**")

        for i, row in enumerate(table_rows):
            if row["status"] == "✔ PASS":
                bg = "#1a4731"
            elif row["status"] == "✗ FAIL":
                bg = "#5c1515"
            else:
                bg = "#33312a"

            cols = st.columns(COL_W)
            for col, val in zip(cols[:5], [row["status"], row["field"],
                                            row["expected"], row["found"], row["similarity"]]):
                col.markdown(
                    f"<div style='background:{bg};color:#f0f0f0;padding:6px 8px;"
                    f"border-radius:4px;margin-bottom:4px'>{val}</div>",
                    unsafe_allow_html=True,
                )
            # with cols[5]:
            #     # View option intentionally hidden; keep _view_in_gad_dialog and
            #     # render_highlight wired so it can be restored later if needed.
            #     if row["gad_ref"]:
            #         if st.button("🔍 View", key=f"viewbtn_{rep.sub_pdf}_{i}", use_container_width=True):
            #             view_label = row["field"] if row["field"] != "-" else f"value: {row['found']}"
            #             _view_in_gad_dialog(row["gad_ref"], view_label)
            #     else:
            #         st.markdown(
            #             "<div style='color:#888;padding:6px 8px'>n/a</div>",
            #             unsafe_allow_html=True,
            #         )

        if rep.overall_pass:
            st.success("All fields validated successfully.")
        else:
            failed = [r.field for r in rep.results if not r.match]
            if failed:
                st.error(f"Failed fields: {', '.join(failed)}")

        if rep.orphan_values:
            st.caption(
                f"🔎 {len(rep.orphan_values)} additional value(s) found in the "
            "drawing with no matching label — matched against the main GAD above."
            )

        if debug_mode and rep.sub_kv:
            with st.expander("Extracted KV fields (debug)"):
                kv_df = pd.DataFrame(rep.sub_kv.items(), columns=["Key", "Value"])
                st.dataframe(kv_df, use_container_width=True, hide_index=True)

# Raw log
with st.expander("🔍 Raw log output"):
    st.code(_strip(res.get("log", "")), language=None)
