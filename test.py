"""
PDF Cross-Validator v5  —  Targeted Gemini-on-FAIL
────────────────────────────────────────────────────────────────
Pipeline per sub-PDF:

  1. pdfplumber   → raw text + table rows          (always, free)
  2. regex        → extract KV fields              (always, free)
  3. validate()   → compare against BOM            (always, free)
  4. Gemini       → ONLY for fields that FAILED    (conditional)
       • Targeted prompt: "find me SIZE in this text"
       • Re-validates only the failed fields
       • Upgrades FAIL→PASS if Gemini finds a match

Gemini call budget:
  • Main PDF  : 1 call if BOM < 2 items OR spec < 5 fields
  • Sub-PDF   : 1 call per sub-PDF that has ANY failed field
  → Clean PDFs  : 0 Gemini calls
  → Worst case  : 1 + N calls (one per PDF)

Gemini key: --gemini-key  OR  env var GEMINI_API_KEY
────────────────────────────────────────────────────────────────
"""

import json, re, sys, os, argparse, io
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import pdfplumber
from rapidfuzz import fuzz, process as fz_process

# ── colour helpers ────────────────────────────────────────────
G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; B="\033[1m"; X="\033[0m"
ok  = lambda s: f"{G}✔ {s}{X}"
err = lambda s: f"{R}✗ {s}{X}"
warn= lambda s: f"{Y}⚠ {s}{X}"
hdr = lambda s: f"{B}{C}{s}{X}"
inf = lambda s: f"{C}ℹ {s}{X}"

MIN_BOM_ITEMS  = 2   # below this → Gemini used for main PDF BOM
MIN_SPEC_FIELDS = 5  # below this → Gemini used for main PDF spec

# ── data structures ───────────────────────────────────────────
@dataclass
class BOMItem:
    sr_no:str=""; description:str=""; quantity:str=""
    material:str=""; size:str=""; remarks:str=""
    gad_row_ref:Optional[dict]=None   # {"page":int, "bbox":[x0,top,x1,bottom]} in the GAD

@dataclass
class ValidationResult:
    field:str; main_value:str; sub_value:str
    match:bool; similarity:float
    note:str=""; gemini_resolved:bool=False
    gad_ref:Optional[dict]=None       # where main_value is shown in the GAD

@dataclass
class OrphanValue:
    """A value found in the sub-PDF that never paired with a recognized label."""
    value:str
    gad_ref:Optional[dict]=None

@dataclass
class SubPDFReport:
    sub_pdf:str
    matched_bom_item:Optional[BOMItem]
    results:list = field(default_factory=list)
    overall_pass:bool=False
    sub_kv:dict = field(default_factory=dict)
    gemini_used:bool=False
    orphan_values:list = field(default_factory=list)

# ── STEP 1: raw extraction ────────────────────────────────────

def extract_raw(path:str) -> tuple[str, list[list]]:
    full_text, rows = [], []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            full_text.append(t)
            for table in page.extract_tables():
                for row in table:
                    cleaned = [str(c).strip() if c else "" for c in row]
                    if any(cleaned):
                        rows.append(cleaned)
    return "\n".join(full_text), rows

# ── STEP 1b: GAD word-position index (for "View in GAD" highlight) ──

def index_words(path:str) -> list[dict]:
    """Every word in the GAD with its page + bounding box, used to locate
    where a BOM/field value (or a label-less orphan value) is shown.
    Some CAD-exported PDFs carry off-canvas leftover content (negative or
    out-of-page coordinates, sometimes even mirrored text) -- skip anything
    outside the visible page so it can never be matched or highlighted."""
    words = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for w in page.extract_words():
                if w["x0"] < -1 or w["top"] < -1 or \
                   w["x1"] > page.width + 1 or w["bottom"] > page.height + 1:
                    continue
                words.append({
                    "page": pno, "text": w["text"],
                    "x0": w["x0"], "top": w["top"],
                    "x1": w["x1"], "bottom": w["bottom"],
                })
    return words

def _row_words(gad_words:list[dict], page:int, top:float, tol:float=2.0) -> list[dict]:
    return sorted(
        (w for w in gad_words if w["page"] == page and abs(w["top"] - top) <= tol),
        key=lambda w: w["x0"]
    )

def _row_bbox(words:list[dict]) -> list[float]:
    return [min(w["x0"] for w in words), min(w["top"] for w in words),
            max(w["x1"] for w in words), max(w["bottom"] for w in words)]

def locate_bom_row(sr_no:str, description:str, gad_words:list[dict]) -> Optional[dict]:
    """Anchor a BOM row to its location in the GAD: find the sr_no immediately
    followed, on the same line, by the first word of its description."""
    if not description or not sr_no:
        return None
    first_word = description.split()[0].upper()
    sr_target  = _code_norm(sr_no)
    for w in gad_words:
        if not w["text"].strip().isdigit() or _code_norm(w["text"]) != sr_target:
            continue
        same_line = [o for o in gad_words
                     if o["page"] == w["page"] and abs(o["top"] - w["top"]) <= 1.5
                     and o["x0"] > w["x0"]]
        same_line.sort(key=lambda o: o["x0"])
        if same_line and same_line[0]["text"].upper() == first_word and \
           same_line[0]["x0"] - w["x1"] < 100:
            return {"page": w["page"], "bbox": _row_bbox(_row_words(gad_words, w["page"], w["top"]))}
    return None

def find_field_bbox(field_value:str, row_ref:Optional[dict], gad_words:list[dict]) -> Optional[dict]:
    """Find the exact cell for a field's value within its BOM row's line.
    Falls back to the whole-row bbox if the value can't be isolated.

    Purely-numeric targets (e.g. QUANTITY="1") are matched against a single
    whole word only -- never a substring spanning a word-concatenation join
    (e.g. sr_no "11" + "TERMINAL" + "BOX" concatenates to "...BOX", and the
    *next* cell "1" would otherwise look like a clean suffix match of that
    string purely by coincidence of where the words happen to join)."""
    if not row_ref or not field_value:
        return row_ref
    target = _code_norm(field_value)
    if not target:
        return row_ref
    page, (x0, top, x1, bottom) = row_ref["page"], row_ref["bbox"]
    row = _row_words(gad_words, page, top)

    if target.isdigit():
        for w in row:
            if _code_norm(w["text"]) == target:
                return {"page": page, "bbox": [w["x0"], w["top"], w["x1"], w["bottom"]]}
        return row_ref

    # The leftmost cell is always the sr_no column -- never part of a text
    # field's value, so excluding it keeps e.g. MATERIAL/DESCRIPTION boxes
    # from bleeding into the sr_no cell when the row starts with a digit.
    search_row = row[1:] if row and row[0]["text"].strip().isdigit() else row
    for i in range(len(search_row)):
        acc = ""
        for j in range(i, min(i + 6, len(search_row))):
            acc += _code_norm(search_row[j]["text"])
            if acc == target or target in acc:
                return {"page": page, "bbox": _row_bbox(search_row[i:j + 1])}
            if len(acc) > len(target) + 10:
                break
    return row_ref

def _num_norm(s:str) -> str:
    """Keep only digits and the decimal point -- orphan values are numeric/
    dimensional, so letters must never let e.g. '125' match inside '3125W'."""
    return re.sub(r'[^0-9.]', '', s)

def find_value_anywhere(value:str, gad_words:list[dict]) -> Optional[dict]:
    """Locate a label-less orphan value verbatim anywhere in the GAD.
    Prefers an exact normalized match; falls back to a boundary-safe
    substring match (rejects matches glued to another digit, e.g. '125'
    inside '3125', so we don't highlight an unrelated number)."""
    target = _num_norm(value)
    if len(target) < 2:
        return None
    for w in gad_words:
        if _num_norm(w["text"]) == target:
            return {"page": w["page"], "bbox": [w["x0"], w["top"], w["x1"], w["bottom"]]}
    best, best_len = None, None
    for w in gad_words:
        wn = _num_norm(w["text"])
        idx = wn.find(target)
        if idx == -1:
            continue
        before = wn[idx - 1] if idx > 0 else ""
        after  = wn[idx + len(target)] if idx + len(target) < len(wn) else ""
        if before.isdigit() or after.isdigit():
            continue
        if best_len is None or len(wn) < best_len:
            best, best_len = w, len(wn)
    if best:
        return {"page": best["page"], "bbox": [best["x0"], best["top"], best["x1"], best["bottom"]]}
    return None

def render_highlight(gad_pdf, ref:dict, crop_pad:float=180, full_page:bool=False,
                      resolution:int=150) -> bytes:
    """Render the GAD page at `ref` with a translucent highlighter-style
    overlay on the bbox (text stays readable underneath, unlike an opaque
    box). Returns PNG bytes for direct display.
    `gad_pdf` may be a file path or a file-like object (e.g. io.BytesIO).
    `full_page=True` renders the entire page instead of a cropped window
    around the bbox -- lets the UI offer pan/zoom over the whole drawing."""
    page_no = ref["page"]
    x0, top, x1, bottom = ref["bbox"]
    pad = 1.5   # slight overflow past the text edges, like a real highlighter
    hl_box = (x0 - pad, top - pad, x1 + pad, bottom + pad)
    with pdfplumber.open(gad_pdf) as pdf:
        page = pdf.pages[page_no - 1]
        if full_page:
            target = page
        else:
            crop_box = (
                max(0, x0 - crop_pad), max(0, top - crop_pad),
                min(page.width, x1 + crop_pad), min(page.height, bottom + crop_pad),
            )
            target = page.crop(crop_box)
        im = target.to_image(resolution=resolution)
        im.draw_rect(hl_box, stroke=(255, 200, 0, 220), stroke_width=1,
                     fill=(255, 255, 0, 90))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

def attach_bom_refs(bom:list[BOMItem], gad_words:list[dict]) -> None:
    for item in bom:
        item.gad_row_ref = locate_bom_row(item.sr_no, item.description, gad_words)

# ── STEP 1c: orphan (label-less) value detection ────────────────

_ORPHAN_NUM_RE = re.compile(r'(?<![\w./])(?:\d{1,4}\.\d{1,3}|\d{2,4})(?![\w./])')

def extract_orphan_values(sub_text:str, sub_kv:dict) -> list[str]:
    """Standalone numeric/dimensional tokens in the sub-PDF that never made
    it into a labeled KV pair -- e.g. a bare dimension callout drawn directly
    on the component view with no adjacent key (like the '12.5'/'0.1' that
    make up an unlabeled 'O 12.5 +/- 0.1' size callout)."""
    known_values = " | ".join(sub_kv.values())
    seen, orphans = set(), []
    for m in _ORPHAN_NUM_RE.finditer(sub_text):
        tok = m.group(0)
        if tok in seen or tok in known_values:
            continue
        seen.add(tok)
        orphans.append(tok)
    return orphans

# ── STEP 2: regex KV extraction ───────────────────────────────

KV_PATTERN = re.compile(
    r"([A-Z][A-Z0-9 \(\)/\-\.]{2,55}?)\s*[:\|]\s*([^\n:]{1,150})",
    re.MULTILINE
)

_NOISE_RE = re.compile(
    r"^\d[\d\.\-\s]*$"
    r"|^(UNLESS|REMOVE|MACHINE|SURFACE|TOLERANCES|THESE|ALL DIMENSIONS)"
    r"|(IS\s*:\s*210)"
    r"|^\s*$",
    re.I
)

# Field aliases — defined early so _ALL_KNOWN_KEYS can reference them
BOM_FIELD_ALIASES = {
    "description": ["TITLE","DRAWING TITLE","ITEM","COMPONENT","PART NAME",
                    "DESCRIPTION","EQUIPMENT","PART","NAME","COMPONENT NAME"],
    "quantity":    ["QUANTITY","QTY","NOS","NUMBER","COUNT","NO OF","NO"],
    "material":    ["MATERIAL","MAT","GRADE","ALLOY","SPEC","MAT SPEC","MATERIAL SPEC"],
    "size":        ["SIZE","DIMENSION","DIMENSIONS","RATING","FLANGE RATING",
                    "NOMINAL SIZE","FLANGE SIZE","OD","DN","CLASS","BOX SIZE",
                    "TERMINAL BOX SIZE","OVERALL SIZE"],
    "remarks":     ["STANDARD","CERT","CERTIFICATION","EN","ASTM","IS STANDARD",
                    "REVISION","NORM","TEST CERT","MATERIAL CERT","APPROVAL"],
}

# Flat set of every known alias — used to detect "value is actually a field name" misreads
_ALL_KNOWN_KEYS: set[str] = {
    alias.upper()
    for aliases in BOM_FIELD_ALIASES.values()
    for alias in aliases
}

def _is_label(s:str) -> bool:
    s = s.strip()
    return 2 <= len(s) <= 70 and not s.isdigit() and bool(re.search(r"[A-Za-z]", s))

def _is_value(s:str) -> bool:
    """True if string is a real value — not noise, not a separator, not another field name."""
    s = s.strip()
    if not s:
        return False
    if _NOISE_RE.search(s):
        return False
    if len(s) == 1 and not s.isalnum():
        return False
    # Key insight: if the "value" is itself a known field name, the row was a section
    # header misread as KEY: VALUE  (e.g. "TERMINAL BOX: SIZE" → drop it)
    if s.upper() in _ALL_KNOWN_KEYS:
        return False
    return True

def _pick_value_from_row(cells:list[str]) -> str:
    """Return first real value cell, skipping separator cells."""
    for c in cells:
        c = c.strip()
        if c in (":", "|", "-", "–", "="):
            continue
        if _is_value(c):
            return c
    return ""

def regex_extract_kv(text:str, rows:list[list]) -> dict:
    kv = {}

    # From raw text
    for m in KV_PATTERN.finditer(text):
        k = m.group(1).strip().upper()
        v = m.group(2).strip()
        if len(k) < 3 or len(k) > 70 or k.isdigit():
            continue
        if not v or _NOISE_RE.search(v):
            continue
        if v.upper() in _ALL_KNOWN_KEYS:   # section-header misread guard
            continue
        kv[k] = v

    # From table rows — handles 2-col, 3-col+separator, merged cells, label-then-value
    for idx, row in enumerate(rows):
        ne = [str(c).strip() for c in row if str(c).strip() and len(str(c).strip()) > 1]
        if not ne:
            continue

        first = ne[0].strip()
        KEY   = first.upper()
        if not _is_label(first) or len(KEY) > 70:
            continue

        # Case 1: value in same row (handles 2-col, 3-col with ":" cell, multi-col)
        val = _pick_value_from_row(ne[1:])
        if val:
            kv.setdefault(KEY, val)
            continue

        # Case 2: newline-merged cell "SIZE\n500 x 500 x 210"
        if "\n" in first:
            parts = [p.strip() for p in first.split("\n") if p.strip()]
            if len(parts) >= 2 and _is_label(parts[0]) and _is_value(parts[1]):
                kv.setdefault(parts[0].upper(), parts[1])
                continue

        # Case 3: value is on the next row (label row alone, then value row)
        if len(ne) == 1 and idx + 1 < len(rows):
            next_ne = [str(c).strip() for c in rows[idx+1]
                       if str(c).strip() and len(str(c).strip()) > 1]
            if next_ne and not _is_label(next_ne[0]):
                kv.setdefault(KEY, next_ne[0].strip())

    return kv

def parse_bom_regex(rows:list[list]) -> list[BOMItem]:
    items = []
    for row in rows:
        ne = [c for c in row if c]
        if len(ne) < 3:
            continue
        first = ne[0].split("\n")[0].strip()
        if not re.match(r"^\d+[\+\d\-]*$", first):
            continue
        max_lines = max(len(c.split("\n")) for c in ne[:6])
        padded = [(c.split("\n") + [""] * max_lines)[:max_lines]
                  for c in (ne + [""] * 6)[:6]]
        for i in range(max_lines):
            vals = [col[i].strip() for col in padded]
            if vals[0] and re.match(r"^\d+[\+\d\-]*$", vals[0]):
                items.append(BOMItem(
                    sr_no=vals[0], description=vals[1].upper(),
                    quantity=vals[2], material=vals[3].upper(),
                    size=vals[4], remarks=vals[5]
                ))
    return items

# ── STEP 3: Gemini (conditional) ─────────────────────────────

GEMINI_MODEL   = "gemini-2.5-flash"
_gemini_client = None

def _init_gemini(api_key:str) -> bool:
    global _gemini_client
    if _gemini_client:
        return True
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        print(inf(f"Gemini ready ({GEMINI_MODEL})"))
        return True
    except Exception as e:
        print(warn(f"Gemini init failed: {e}"))
        return False

def _get_api_key(cli_key:str) -> str:
    return cli_key or os.environ.get("GEMINI_API_KEY", "")

def gemini_extract_main(text:str, existing_kv:dict, existing_bom:list) -> tuple[dict, list]:
    """Called only when regex gave a weak main-PDF result."""
    if not _gemini_client:
        return existing_kv, existing_bom

    have_keys = list(existing_kv.keys())[:20]
    have_bom  = len(existing_bom)

    prompt = f"""You are parsing an engineering GAD/BOM drawing PDF.

Already extracted spec keys (skip these): {have_keys}
BOM rows already found: {have_bom}

Tasks:
1. Extract any spec key-value pairs NOT in the list above → "kv": {{"KEY": "value"}}
2. {"Extract the BOM table → \"bom\": [{{sr_no, description, quantity, material, size, remarks}}, ...]" if have_bom < MIN_BOM_ITEMS else "\"bom\": []  (BOM already found, skip)"}

Return ONLY valid JSON. No markdown. No explanation.
Format: {{"kv": {{}}, "bom": []}}

RAW TEXT:
{text[:5000]}"""

    try:
        raw = _gemini_client.generate_content(prompt).text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        gem_kv = {k.upper().strip(): str(v).strip()
                  for k, v in result.get("kv", {}).items() if k and v}
        merged_kv = {**gem_kv, **existing_kv}   # regex wins on conflict

        if result.get("bom") and have_bom < MIN_BOM_ITEMS:
            bom = [BOMItem(
                sr_no      = str(r.get("sr_no","")).strip(),
                description= str(r.get("description","")).upper().strip(),
                quantity   = str(r.get("quantity","")).strip(),
                material   = str(r.get("material","")).upper().strip(),
                size       = str(r.get("size","")).strip(),
                remarks    = str(r.get("remarks","")).strip(),
            ) for r in result["bom"] if isinstance(r, dict)]
        else:
            bom = existing_bom

        return merged_kv, bom
    except Exception as e:
        print(warn(f"  Gemini main error: {e}"))
        return existing_kv, existing_bom


def gemini_fix_failed_fields(
    sub_text: str,
    bom_item: BOMItem,
    failed_results: list[ValidationResult]
) -> dict[str, str]:
    """
    Targeted Gemini call: given only the FAILED fields, ask Gemini to find
    their values in the sub-PDF text.

    Returns {FIELD_NAME: found_value} for any it could locate.
    e.g. {"SIZE": "500 x 500 x 210"}
    """
    if not _gemini_client:
        return {}

    # Build a clear list of what we need
    needed = []
    for r in failed_results:
        aliases = BOM_FIELD_ALIASES.get(r.field.lower(), [r.field])
        needed.append({
            "field":          r.field,
            "expected_value": r.main_value,
            "look_for_keys":  aliases[:4],   # top aliases only to keep prompt tight
        })

    prompt = f"""You are reading an engineering drawing PDF for component: {bom_item.description}

I already validated this drawing against a Bill of Materials and some fields FAILED
because the values could not be found by automated parsing.

For each item below, search the raw PDF text and find the actual value.
Return the value EXACTLY as it appears in the drawing — do not paraphrase.

Fields to find:
{json.dumps(needed, indent=2)}

Rules:
- Return ONLY valid JSON: {{"FIELD_NAME": "found_value", ...}}
- Use the exact FIELD_NAME from the input (e.g. "SIZE", "MATERIAL")
- If you genuinely cannot find a field, omit it — do not guess
- No markdown, no explanation

RAW PDF TEXT:
{sub_text[:4000]}"""

    try:
        raw = _gemini_client.generate_content(prompt).text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return {k.upper().strip(): str(v).strip() for k, v in result.items() if k and v}
    except Exception as e:
        print(warn(f"  Gemini fix error: {e}"))
        return {}

# ── STEP 4: BOM matching ──────────────────────────────────────

def match_bom(sub_pdf_name:str, bom:list[BOMItem]) -> tuple[Optional[BOMItem], float]:
    stem = Path(sub_pdf_name).stem.upper()
    stem = re.sub(r"^\d+[_\-]", "", stem).replace("_"," ").strip()
    if not bom:
        return None, 0.0
    best, score = None, 0.0
    for item in bom:
        s = fuzz.token_sort_ratio(stem, item.description)
        if s > score:
            score, best = s, item
    return (best, score) if score >= 35 else (None, score)

# ── STEP 5: validation ────────────────────────────────────────

def _norm(s:str) -> str:
    return re.sub(r"[\s\.\-]+", " ", s.upper().strip())

# ── strict / code-aware similarity ───────────────────────────

# Purely numeric + dimensional separators (x, ×, /)
_DIM_RE = re.compile(r'^[\d\s×xX\/]+$')

def _plain_size_numbers(s:str) -> Optional[list[float]]:
    """Return ordered numeric size/tolerance tokens when letters are only
    dimensional separators. This treats 'Ø12.5±0.1' and '12.5 0.1' as the
    same value without making codes like 'M10' equal to plain '10'."""
    cleaned = s.upper()
    cleaned = cleaned.replace("Ø", " ").replace("⌀", " ").replace("Φ", " ")
    cleaned = re.sub(r'(?<![A-Z])O\s*(?=\d)', ' ', cleaned)  # OCR/plain O for diameter
    cleaned = cleaned.replace("±", " ").replace("+/-", " ")
    cleaned = cleaned.replace("×", "X")
    if re.search(r'[A-WY-Z]', cleaned):  # allow X only as a dimension separator
        return None
    numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', cleaned)]
    return numbers or None

def _is_dimensional(s:str) -> bool:
    s = s.strip()
    return bool(s) and bool(_DIM_RE.match(s)) and bool(re.search(r'\d', s))

def _dim_sim(a:str, b:str) -> float:
    """Ordered dimensional comparison. Position matters: 50x500x210 ≠ 500x50x210.
    Extracts numbers as floats so '500.0 x 500.0' == '500 x 500'."""
    da = _plain_size_numbers(a)
    db = _plain_size_numbers(b)
    if da is None or db is None:
        na, nb = _norm(a), _norm(b)
        da = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', na)]
        db = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', nb)]
    if not da or not db:
        return 0.0
    if da == db:
        return 100.0
    # Dimensions differ — cap below pass threshold, show partial score for context
    matches = sum(1 for x, y in zip(da, db) if x == y)
    total   = max(len(da), len(db))
    return min((matches / total) * 100.0, 60.0)

def _code_norm(s:str) -> str:
    """Strip ALL non-alphanumeric chars: 'EN 10204-3.1' → 'EN1020431', 'SS 304' → 'SS304'."""
    return re.sub(r'[^A-Z0-9]', '', s.upper())

def _code_sim(a:str, b:str) -> float:
    """Exact match after stripping separators. Spaces/hyphens/dots are ignored;
    actual alphanumeric content must be identical. Non-matching codes are capped
    at 60 (below the 72 pass threshold) so they always fail."""
    na, nb = _code_norm(a), _code_norm(b)
    if na == nb:
        return 100.0
    return min(fuzz.ratio(na, nb), 60.0)

def _val_sim(a:str, b:str) -> float:
    """Fuzzy token similarity — used only for DESCRIPTION."""
    return fuzz.token_set_ratio(_norm(a), _norm(b))

# Unit suffixes that appear after a quantity number and carry no numeric meaning
_QTY_UNIT_RE = re.compile(
    r'\s*(NOS?\.?|PCS\.?|PIECES?|EA\.?|EACH|SETS?|UNITS?|OFF|NO\.?)\s*$', re.I
)

def _qty_sim(a:str, b:str) -> float:
    """Compare quantities after stripping trailing unit words.
    '1 NOS' == '1', '3 PCS' == '3', but '1' != '2'."""
    def _strip(s):
        return _code_norm(_QTY_UNIT_RE.sub('', s.strip()))
    na, nb = _strip(a), _strip(b)
    if na == nb:
        return 100.0
    return min(fuzz.ratio(na, nb), 60.0)

# Fields that must match exactly (after stripping formatting separators)
_STRICT_FIELDS = {"MATERIAL", "REMARKS"}

def _field_sim(field:str, a:str, b:str) -> float:
    f = field.upper()
    if f == "SIZE":
        da = _plain_size_numbers(a)
        db = _plain_size_numbers(b)
        if da is not None and db is not None:
            return _dim_sim(a, b)
        na, nb = _norm(a), _norm(b)
        if _is_dimensional(na) and _is_dimensional(nb):
            return _dim_sim(a, b)
        return _code_sim(a, b)   # non-numeric size codes like DN500, CLASS 150
    if f == "QUANTITY":
        return _qty_sim(a, b)
    if f in _STRICT_FIELDS:
        return _code_sim(a, b)
    return _val_sim(a, b)        # DESCRIPTION: keep fuzzy

def _field_match(field:str, a:str, b:str, threshold:int=72) -> tuple[bool,float]:
    sim = _field_sim(field, a, b)
    return sim >= threshold, sim

def _find_in_kv(query:str, kv:dict, threshold=45) -> tuple[str,str,float]:
    if not kv:
        return "","",0.0
    r = fz_process.extractOne(query.upper(), list(kv.keys()), scorer=fuzz.token_sort_ratio)
    if r and r[1] >= threshold:
        return r[0], kv[r[0]], r[1]
    return "","",0.0

def validate(bom_item:BOMItem, sub_kv:dict, gad_words:Optional[list[dict]]=None) -> list[ValidationResult]:
    results = []
    bom_values = {
        "description": bom_item.description,
        "quantity":    bom_item.quantity,
        "material":    bom_item.material,
        "size":        bom_item.size,
        "remarks":     bom_item.remarks,
    }
    for field_name, bom_val in bom_values.items():
        if not bom_val:
            continue
        aliases = list(BOM_FIELD_ALIASES.get(field_name, [field_name.upper()]))
        if field_name == "description":
            aliases = [bom_item.description] + aliases

        best_key, best_val, best_key_score = "", "", 0.0
        for alias in aliases:
            k, v, ks = _find_in_kv(alias, sub_kv)
            if ks > best_key_score:
                best_key, best_val, best_key_score = k, v, ks

        gad_ref = (find_field_bbox(bom_val, bom_item.gad_row_ref, gad_words)
                   if gad_words else None)

        if not best_val:
            results.append(ValidationResult(
                field=field_name.upper(), main_value=bom_val,
                sub_value="— NOT FOUND —", match=False, similarity=0.0,
                note="Field not found in sub-PDF", gad_ref=gad_ref
            ))
            continue

        if field_name == "description":
            sk = _val_sim(bom_val, best_key)
            sv = _val_sim(bom_val, best_val)
            sim, used_val = (sk, best_key) if sk >= sv else (sv, best_val)
            match = sim >= 72
        else:
            match, sim = _field_match(field_name, bom_val, best_val)
            used_val = best_val

        results.append(ValidationResult(
            field=field_name.upper(), main_value=bom_val,
            sub_value=used_val, match=match, similarity=sim,
            note=f"matched key='{best_key}' (key_score={best_key_score:.0f})",
            gad_ref=gad_ref
        ))
    return results


def gemini_retry_failed(
    results: list[ValidationResult],
    bom_item: BOMItem,
    sub_text: str
) -> tuple[list[ValidationResult], bool]:
    """
    Takes the initial validation results, calls Gemini for FAILed fields only,
    re-validates those fields, and returns the upgraded result list + whether
    Gemini was actually called.
    """
    failed = [r for r in results if not r.match]
    if not failed:
        return results, False   # nothing to fix

    print(warn(f"  {len(failed)} field(s) failed regex validation — "
               f"calling Gemini to locate: "
               f"{[r.field for r in failed]}"))

    gem_values = gemini_fix_failed_fields(sub_text, bom_item, failed)
    if not gem_values:
        print(warn("  Gemini returned no values"))
        return results, True   # called but got nothing

    print(inf(f"  Gemini found: {gem_values}"))

    # Rebuild result list: upgrade any FAIL whose field Gemini found
    upgraded = []
    for r in results:
        if r.match or r.field not in gem_values:
            upgraded.append(r)
            continue

        gem_val = gem_values[r.field]
        match, sim = _field_match(r.field, r.main_value, gem_val)
        upgraded.append(ValidationResult(
            field         = r.field,
            main_value    = r.main_value,
            sub_value     = gem_val,
            match         = match,
            similarity    = sim,
            note          = f"resolved by Gemini (regex had: '{r.sub_value}')",
            gemini_resolved = True,
            gad_ref       = r.gad_ref,
        ))
        status = ok("PASS") if match else err("FAIL")
        print(f"  Gemini re-check {r.field}: {status} "
              f"({r.main_value!r} vs {gem_val!r}, sim={sim:.0f}%)")

    return upgraded, True


def gemini_final_judge(
    sub_text: str,
    bom_item: BOMItem,
    results: list[ValidationResult],
) -> tuple[list[ValidationResult], bool]:
    """
    Final-pass Gemini review over ALL validation results for one sub-PDF.
    Acts as a senior engineer who reads the raw drawing and gives the
    authoritative verdict on every field — overriding both false positives
    and false negatives from automated matching.
    Returns the (possibly corrected) results and whether it was called.
    """
    if not _gemini_client:
        return results, False

    # Build a readable summary of what the automated checks found
    field_rows = []
    for r in results:
        status = "PASS" if r.match else "FAIL"
        field_rows.append(
            f"  {r.field:<14} | BOM: {r.main_value:<30} | Drawing: {r.sub_value:<30} | {status} (sim={r.similarity:.0f}%)"
        )
    fields_table = "\n".join(field_rows)

    prompt = f"""You are a senior mechanical engineer performing the final quality-assurance check on an automated drawing validation system.

COMPONENT BEING VALIDATED: {bom_item.description}

BILL OF MATERIALS — EXPECTED VALUES (ground truth):
  Description : {bom_item.description}
  Quantity    : {bom_item.quantity}
  Material    : {bom_item.material}
  Size        : {bom_item.size}
  Remarks     : {bom_item.remarks}

AUTOMATED VALIDATION RESULTS (what the system found in the drawing):
{fields_table}

RAW DRAWING TEXT (for your reference — search here for the actual values):
{sub_text[:4000]}

────────────────────────────────────────────────────────────────
YOUR TASK: For every field above, give the authoritative PASS or FAIL verdict.
Use the raw drawing text to confirm what is actually written there.

EQUIVALENCE RULES YOU MUST APPLY:
1. QUANTITY  — Unit suffixes are irrelevant: "1 NOS", "1 No.", "ONE (1)" all equal "1".
               Only the numeric count matters.
2. STANDARDS — Separator formatting is irrelevant: "EN 10204-3.1" == "EN10204-3.1" == "EN102043.1".
               Spaces, hyphens, and dots between alphanumerics are ignored.
               But "3.1" ≠ "3.2" — revision numbers must match.
3. MATERIAL  — Grade codes are EXACT: "SS304" ≠ "SS301" ≠ "SS316". Every digit matters.
               Spaces/hyphens are ignored: "SA-213 TP304" == "SA213TP304".
4. SIZE      — ALL dimensions AND their ORDER must match exactly (they define physical axes):
               "500 x 500 x 210" ≠ "500 x 50 x 210" (wrong number).
               "50 x 500 x 210"  ≠ "500 x 50 x 210" (wrong order).
               Separator style is irrelevant: "500x500x210" == "500 x 500 x 210".
               Diameter/tolerance symbols are formatting only:
               "Ø12.5±0.1" == "12.5 0.1" == "O12.5 +/- 0.1".
5. DESCRIPTION — Partial / abbreviated matches are acceptable if they clearly refer
               to the same component (e.g. "TERMINAL BOX" matches "TB", "TERM. BOX").

IMPORTANT:
- If the raw drawing text confirms a value that the automated system MISSED or mis-matched,
  set the correct verdict and quote the exact value from the drawing.
- If the automated system gave PASS but the values are actually different, set FAIL.
- If a field is genuinely absent from the drawing, keep FAIL with found_value = "NOT FOUND".
- Do NOT guess. If you cannot find a field in the raw text, say so.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "FIELD_NAME": {{
    "verdict": "PASS" or "FAIL",
    "found_value": "exact value as written in the drawing, or NOT FOUND",
    "reason": "one concise line"
  }},
  ...
}}
Include every field from the table above. Use the exact FIELD_NAME (e.g. "SIZE", "MATERIAL").
"""

    try:
        raw = _gemini_client.generate_content(prompt).text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        verdicts = json.loads(raw)
    except Exception as e:
        print(warn(f"  Gemini final judge error: {e}"))
        return results, True

    print(inf("  Gemini final judge verdicts:"))
    corrected = []
    for r in results:
        v = verdicts.get(r.field)
        if not v or not isinstance(v, dict):
            corrected.append(r)
            continue

        new_verdict  = str(v.get("verdict", "")).upper().strip()
        found_val    = str(v.get("found_value", r.sub_value)).strip()
        reason       = str(v.get("reason", "")).strip()
        gem_pass     = new_verdict == "PASS"

        # Re-score: if Gemini says PASS, sim=100; if FAIL, keep existing sim (or 0)
        new_sim  = 100.0 if gem_pass else min(r.similarity, 60.0)
        changed  = gem_pass != r.match

        tag = ""
        if changed:
            tag = f"  {'PASS->FAIL' if r.match else 'FAIL->PASS'} ({reason})"

        status_str = ok("PASS") if gem_pass else err("FAIL")
        print(f"  Judge {r.field}: {status_str}"
              + (f"{Y}{tag}{X}" if changed else f"  ({reason[:60]})"))

        corrected.append(ValidationResult(
            field           = r.field,
            main_value      = r.main_value,
            sub_value       = found_val if found_val != "NOT FOUND" else r.sub_value,
            match           = gem_pass,
            similarity      = new_sim,
            note            = f"final-judge: {reason}",
            gemini_resolved = True,
            gad_ref         = r.gad_ref,
        ))

    return corrected, True


# ── STEP 6: report ────────────────────────────────────────────

def print_report(spec:dict, bom:list[BOMItem], reports:list[SubPDFReport],
                 gemini_calls:int):
    print(f"\n{hdr('='*66)}")
    print(hdr("  PDF CROSS-VALIDATION REPORT"))
    print(hdr('='*66))

    print(f"\n{hdr(f'MAIN PDF — Equipment Spec ({len(spec)} fields)')}")
    for k, v in list(spec.items())[:35]:
        if len(k) < 60:
            print(f"  {C}{k:<42}{X} {v[:60]}")

    print(f"\n{hdr(f'MAIN PDF — BOM ({len(bom)} items)')}")
    for b in bom:
        print(f"  [{b.sr_no:>3}] {b.description:<35} qty={b.quantity:<6} "
              f"mat={b.material:<15} size={b.size}")

    for rep in reports:
        gem_tag = (f" {Y}[Gemini used]{X}" if rep.gemini_used
                   else f" {G}[regex only]{X}")
        print(f"\n{hdr('─'*66)}")
        print(hdr(f"  SUB-PDF: {rep.sub_pdf}") + gem_tag)
        if rep.matched_bom_item:
            b = rep.matched_bom_item
            print(f"  Matched BOM row [{b.sr_no}]: {b.description}")
        else:
            print(warn("  No BOM match found — cannot validate"))

        all_pass = True
        for r in rep.results:
            st  = ok("PASS") if r.match else err("FAIL")
            sc  = G if r.match else (Y if r.similarity > 50 else R)
            tag = f" {Y}[Gemini]{X}" if r.gemini_resolved else ""
            print(f"  {st}  {r.field:<14} "
                  f"MAIN={C}{r.main_value[:38]:<40}{X}"
                  f"SUB={C}{r.sub_value[:38]:<40}{X}"
                  f"sim={sc}{r.similarity:.0f}%{X}{tag}")
            if not r.match:
                all_pass = False
        rep.overall_pass = all_pass
        print(f"\n  {ok('ALL PASSED') if all_pass else err('SOME FAILED')}")

    print(f"\n{hdr('='*66)}")
    total = sum(1 for r in reports if r.overall_pass)
    print(f"  SUMMARY  : {total}/{len(reports)} sub-PDFs fully validated")
    max_calls = 1 + len(reports)
    gem_color = G if gemini_calls == 0 else (Y if gemini_calls <= 2 else R)
    print(f"  {gem_color}Gemini calls: {gemini_calls} "
          f"(0 = all regex; {max_calls} = all Gemini){X}")
    print(hdr('='*66))


def save_json(spec, bom, reports, path, gemini_calls):
    data = {
        "main_spec": spec,
        "main_bom":  [asdict(b) for b in bom],
        "gemini_calls_used": gemini_calls,
        "sub_pdf_reports": [
            {
                "sub_pdf":          r.sub_pdf,
                "matched_bom_item": asdict(r.matched_bom_item) if r.matched_bom_item else None,
                "overall_pass":     r.overall_pass,
                "gemini_used":      r.gemini_used,
                "sub_kv_extracted": r.sub_kv,
                "validation_results": [asdict(v) for v in r.results],
                "orphan_values":    [asdict(o) for o in r.orphan_values],
            }
            for r in reports
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON report → {path}")

# ── main orchestration ────────────────────────────────────────

def run(main_pdf:str, sub_pdfs:list[str],
        output:str="validation_report.json",
        gemini_key:str="", debug:bool=False):

    api_key      = _get_api_key(gemini_key)
    has_key      = bool(api_key)
    gemini_calls = 0

    if not has_key:
        print(warn("No Gemini key — regex-only mode "
                   "(pass --gemini-key or set GEMINI_API_KEY)"))

    # ── Main PDF ──────────────────────────────────────────────
    print(hdr(f"\n[1] Extracting MAIN PDF: {Path(main_pdf).name}"))
    main_text, main_rows = extract_raw(main_pdf)
    main_spec = regex_extract_kv(main_text, main_rows)
    main_bom  = parse_bom_regex(main_rows)
    gad_words = index_words(main_pdf)
    print(f"  → regex: {len(main_spec)} spec fields | {len(main_bom)} BOM items")

    main_needs_gemini = (len(main_bom) < MIN_BOM_ITEMS or
                         len(main_spec) < MIN_SPEC_FIELDS)
    if main_needs_gemini and has_key:
        print(warn(f"  Regex insufficient — calling Gemini for main PDF..."))
        _init_gemini(api_key)
        main_spec, main_bom = gemini_extract_main(main_text, main_spec, main_bom)
        gemini_calls += 1
        print(f"  → after Gemini: {len(main_spec)} spec fields | {len(main_bom)} BOM items")
    elif main_needs_gemini:
        print(warn(f"  Regex weak but no Gemini key — results may be incomplete"))
    else:
        print(inf("  Regex sufficient — Gemini skipped for main PDF"))

    attach_bom_refs(main_bom, gad_words)

    # ── Sub PDFs ──────────────────────────────────────────────
    reports = []
    for i, sp in enumerate(sub_pdfs):
        name = Path(sp).name
        print(hdr(f"\n[{i+2}] Sub-PDF: {name}"))

        sub_text, sub_rows = extract_raw(sp)
        sub_kv = regex_extract_kv(sub_text, sub_rows)
        print(f"  → regex: {len(sub_kv)} fields extracted")

        if debug:
            for k, v in sub_kv.items():
                print(f"     {C}{k}{X}: {v[:80]}")
        else:
            for k, v in list(sub_kv.items())[:6]:
                print(f"     {C}{k}{X}: {v[:60]}")

        # Match to BOM
        matched, score = match_bom(name, main_bom)
        print(f"  → BOM match: "
              f"{matched.description if matched else 'NO MATCH'} (score={score:.0f})")

        # First-pass validation with regex results
        results = validate(matched, sub_kv, gad_words) if matched else []

        # Label-less values: extracted from the drawing but never paired
        # with a recognized key. Only kept if the exact value is echoed
        # somewhere in the main GAD -- otherwise there's no precise spot to
        # point "View" at, so it's dropped rather than shown without one.
        orphan_tokens = extract_orphan_values(sub_text, sub_kv)
        orphan_values = []
        for tok in orphan_tokens:
            ref = find_value_anywhere(tok, gad_words)
            if ref:
                orphan_values.append(OrphanValue(value=tok, gad_ref=ref))
        print(f"  → {len(orphan_values)} label-less value(s) found in drawing "
              f"(also in GAD; {len(orphan_tokens) - len(orphan_values)} skipped, no exact GAD match)")

        # Step A — targeted Gemini retry for failed fields
        gemini_used = False
        if matched and has_key:
            failed_count = sum(1 for r in results if not r.match)
            if failed_count > 0:
                _init_gemini(api_key)
                results, gemini_used = gemini_retry_failed(results, matched, sub_text)
                if gemini_used:
                    gemini_calls += 1
            else:
                print(inf("  All fields passed regex — targeted retry skipped"))
        elif matched and not has_key:
            failed_count = sum(1 for r in results if not r.match)
            if failed_count:
                print(warn(f"  {failed_count} field(s) failed — "
                           f"provide --gemini-key to attempt auto-fix"))

        # Step B — final Gemini judge: comprehensive review of all results
        if matched and results and has_key:
            _init_gemini(api_key)
            print(inf("  Running Gemini final judge on all fields..."))
            results, judge_called = gemini_final_judge(sub_text, matched, results)
            if judge_called:
                gemini_calls += 1
                gemini_used = True

        rep = SubPDFReport(sub_pdf=name, matched_bom_item=matched,
                           results=results, sub_kv=sub_kv,
                           gemini_used=gemini_used, orphan_values=orphan_values)
        reports.append(rep)

    print_report(main_spec, main_bom, reports, gemini_calls)
    save_json(main_spec, main_bom, reports, output, gemini_calls)
    return main_spec, main_bom, reports

# ── path resolver ─────────────────────────────────────────────

def _resolve_paths(raw_args:list[str]) -> list[str]:
    resolved = []
    i = 0
    while i < len(raw_args):
        candidate = raw_args[i]
        if os.path.isfile(candidate):
            resolved.append(candidate); i += 1; continue
        joined = candidate
        j = i + 1; found = False
        while j < len(raw_args):
            joined = joined + " " + raw_args[j]
            if os.path.isfile(joined):
                resolved.append(joined); i = j + 1; found = True; break
            j += 1
        if not found:
            resolved.append(candidate); i += 1
    return resolved

# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PDF Cross-Validator v5 — Targeted Gemini-on-FAIL",
        epilog=(
            "Examples:\n"
            "  python main.py GAD.PDF FLANGE.PDF ELEMENT.PDF\n"
            '  python main.py GAD.PDF "TERMINAL BOX.PDF" --gemini-key AIza...\n'
            "  set GEMINI_API_KEY=AIza... && python main.py GAD.PDF FLANGE.PDF\n\n"
            "Gemini is called ONLY when needed:\n"
            f"  Main PDF  : if BOM < {MIN_BOM_ITEMS} items OR spec < {MIN_SPEC_FIELDS} fields\n"
            "  Sub-PDFs  : one targeted call per sub-PDF that has ≥1 FAIL field\n"
            "  Best case : 0 Gemini calls (all pass regex)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("main_pdf",  help="Path to main PDF (GAD/BOM)")
    ap.add_argument("sub_pdfs",  nargs="+", help="One or more sub-PDF paths")
    ap.add_argument("--output",  default="validation_report.json",
                    help="Output JSON path (default: validation_report.json)")
    ap.add_argument("--gemini-key", default="",
                    help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    ap.add_argument("--debug",   action="store_true",
                    help="Print all extracted KV fields per sub-PDF")
    args = ap.parse_args()

    all_paths = _resolve_paths([args.main_pdf] + args.sub_pdfs)
    main_pdf  = all_paths[0]
    sub_pdfs  = all_paths[1:]

    if not os.path.isfile(main_pdf):
        print(err(f"Main PDF not found: {main_pdf}")); sys.exit(1)
    missing = [p for p in sub_pdfs if not os.path.isfile(p)]
    for m in missing:
        print(warn(f"Sub-PDF not found (skipping): {m}"))
    sub_pdfs = [p for p in sub_pdfs if os.path.isfile(p)]
    if not sub_pdfs:
        print(err("No valid sub-PDFs found.")); sys.exit(1)

    run(main_pdf, sub_pdfs, args.output, args.gemini_key, debug=args.debug)
