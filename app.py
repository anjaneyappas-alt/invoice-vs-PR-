"""
Invoice vs PR Checker
Compares a supplier invoice (photo or PDF) against a Purchase Receive PDF,
line by line, and looks up SKU codes from the Inventory Summary sheet.
Run with:  streamlit run app.py
"""

import base64
import io
import json
import os
import re

import pandas as pd
import pdfplumber
import pypdfium2 as pdfium
import streamlit as st
from anthropic import Anthropic

st.set_page_config(page_title="Invoice vs PR Checker", layout="wide")

MODELS = {
    "Sonnet 5.5 (recommended)": "claude-sonnet-5-5",
    "Opus 5 (most accurate, costlier)": "claude-opus-5",
    "Haiku 5.5 (cheapest)": "claude-haiku-5-5",
}


# ----------------------------------------------------------------------------
# PR PDF parsing (local, no API)
# ----------------------------------------------------------------------------
def parse_pr_pdf(file_bytes):
    """Pull #, description, SKU and received qty out of a Purchase Receive PDF."""
    text_pages = []
    po_number = ""
    pr_number = ""
    vendor = ""

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text_pages.append(page.extract_text() or "")
    full = "\n".join(text_pages)

    m = re.search(r"Purchase Order#\s*:?\s*(\S+)", full)
    if m:
        po_number = m.group(1)
    m = re.search(r"#\s*(PR-\d+)", full)
    if m:
        pr_number = m.group(1)
    m = re.search(r"Vendor Address\s*\n(.+)", full)
    if m:
        vendor = m.group(1).strip()

    # Split into item blocks: a line starting with an item number then text.
    lines = full.split("\n")
    blocks, current = [], None
    for line in lines:
        start = re.match(r"^\s*(\d{1,3})\s{1,}(\S.*)$", line)
        # a real item line also carries an 8-digit HSN or a qty on the same row
        looks_like_item = bool(start) and (
            re.search(r"\b\d{8}\b", line) or re.search(r"[\d,]+\.\d{2}\s*$", line)
        )
        if looks_like_item:
            if current:
                blocks.append(current)
            current = {"no": int(start.group(1)), "text": [start.group(2)]}
        elif current is not None:
            current["text"].append(line.strip())
    if current:
        blocks.append(current)

    rows = []
    for b in blocks:
        body = "\n".join(b["text"])

        sku = ""
        m = re.search(r"SKU\s*:\s*([A-Z0-9\-]+)", body, re.I)
        if m:
            sku = m.group(1).strip()

        qty = None
        m = re.search(r"Rampura\s+([\d,]+\.\d{2})", body)
        if not m:
            m = re.search(r"([\d,]+\.\d{2})", body)
        if m:
            qty = float(m.group(1).replace(",", ""))

        # description = first line(s) before the SKU line, minus HSN/location/qty
        desc_lines = []
        for ln in body.split("\n"):
            if re.match(r"\s*(SKU|Bin|Description)\s*:", ln, re.I):
                break
            desc_lines.append(ln)
        desc = " ".join(desc_lines)
        desc = re.sub(r"\b\d{8}\b.*$", "", desc)          # HSN and everything after
        desc = re.sub(r"\bDC\d+\s+\w+.*$", "", desc)      # location and qty
        desc = re.sub(r"[\d,]+\.\d{2}\s*(Nos|Pcs|Sets)?\s*$", "", desc, flags=re.I)
        desc = re.sub(r"\s{2,}", " ", desc).strip(" ,-")

        if sku or qty is not None:
            rows.append(
                {"PR #": b["no"], "PR SKU": sku, "PR item": desc, "PR qty": qty}
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["PR #"], keep="first").reset_index(drop=True)
    return df, {"po": po_number, "pr": pr_number, "vendor": vendor}


# ----------------------------------------------------------------------------
# Invoice reading (Claude vision)
# ----------------------------------------------------------------------------
INVOICE_PROMPT = """You are reading a supplier tax invoice from India.

Extract EVERY line item exactly as printed. Do not skip lines, do not merge
lines, do not invent lines. Read the quantity column against the row it
actually sits in — on some bills the description text is offset from the
numbers, so align carefully.

Also read any handwriting on the bill (for example "damage", "short",
"not in PO") and put it in the "note" field for that line.

Return ONLY valid JSON, no markdown fence, in this shape:
{
  "invoice_no": "",
  "invoice_date": "",
  "vendor": "",
  "po_number": "",
  "invoice_total": "",
  "printed_total_qty": "",
  "lines": [
    {"sl": 1, "description": "", "part_no": "", "hsn": "",
     "qty": 0, "unit": "Nos", "rate": "", "amount": "", "note": ""}
  ]
}

Rules:
- "qty" must be the number only. Put the unit (Nos, Sets, Mtrs, Coil, Pcs,
  Bdls, Bags, Cartons) in "unit".
- If a value is unreadable, use "" and say so in "note". Never guess a number.
- Keep the lines in the order printed on the bill.
"""


def img_block(data: bytes, media_type: str):
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(data).decode(),
        },
    }


def pdf_pages_to_png(file_bytes, scale=2.0):
    """Render a PDF invoice to PNG pages so the vision model can read it."""
    out = []
    doc = pdfium.PdfDocument(io.BytesIO(file_bytes))
    for i in range(len(doc)):
        pil = doc[i].render(scale=scale).to_pil()
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def call_claude_json(client, model, blocks, system_prompt, max_tokens=8000):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": blocks}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text), resp.usage
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0)), resp.usage
        raise


def read_invoice(client, model, images):
    blocks = [img_block(d, m) for d, m in images]
    blocks.append({"type": "text", "text": "Extract this invoice as JSON."})
    return call_claude_json(client, model, blocks, INVOICE_PROMPT)


# ----------------------------------------------------------------------------
# Matching invoice lines to PR lines (Claude, with the PR as text)
# ----------------------------------------------------------------------------
MATCH_PROMPT = """You match supplier invoice lines to Purchase Receive (PR) lines
for a hardware distributor in India.

You get INVOICE_LINES and PR_LINES as JSON. Match them by product identity:
part numbers, model codes, size, colour and finish. Descriptions are worded very
differently on the two documents (for example invoice "QV6 30 SFD FL/500 IW 20 HIN"
is PR "Hettich Quadro-V6 Drawer Runners ... Length 500mm"), so match on meaning,
not on text similarity.

Unit conversions you must apply before comparing quantities:
- Pipes billed in metres vs received in 3m pieces: metres / 3 = pieces.
- LED strip/rope billed in metres vs received in reels: use the reel length in
  the PR description (5m or 50m) to convert.
- Items billed in pieces vs received in packs: use the pack size in the PR
  description (for example "Pack of 500").
When you convert, say so in the status.

Return ONLY valid JSON, no markdown fence:
{
  "rows": [
    {"inv": 1, "invoice_item": "", "pr_sku": "", "inv_qty": 0, "pr_qty": 0,
     "status": "match" | "short" | "excess" | "not_in_pr" | "not_in_invoice",
     "difference": 0, "note": ""}
  ],
  "summary": ""
}

Rules:
- One row per invoice line, in invoice order. Then add rows for any PR line with
  no invoice line, using "not_in_invoice" and inv_qty 0.
- If an invoice line is a free item (free scheme, free coupling) and the PR has no
  line for it, use status "not_in_pr" and write "free item" in the note.
- Two invoice lines that are the two halves of one kit (for example a sliding
  mechanism and its track) map to the SAME pr_sku: give the first the matched
  quantity and mark the second "kit part of line N" in the note with status
  "match" and difference 0.
- difference = inv_qty - pr_qty after conversion.
- Never invent a SKU. If there is no PR line, leave pr_sku blank.
"""


def match_lines(client, model, invoice_json, pr_df):
    payload = {
        "INVOICE_LINES": invoice_json.get("lines", []),
        "PR_LINES": pr_df.to_dict(orient="records"),
    }
    blocks = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    return call_claude_json(client, model, blocks, MATCH_PROMPT, max_tokens=8000)


# ----------------------------------------------------------------------------
# SKU lookup from the inventory sheet
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_inventory(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    # find the SKU and name columns whatever they are called
    sku_col = next((c for c in df.columns if "sku" in c.lower() or "code" in c.lower()),
                   df.columns[0])
    name_col = next((c for c in df.columns if "name" in c.lower() or "item" in c.lower()
                     or "description" in c.lower()), df.columns[1])
    out = df[[sku_col, name_col]].copy()
    out.columns = ["SKU", "Item"]
    for extra in df.columns:
        if "status" in extra.lower():
            out["Status"] = df[extra]
            break
    return out.dropna(subset=["SKU"]).astype(str)


def candidate_skus(inv_df, text, limit=40):
    """Cheap keyword shortlist so we don't send the whole sheet to the model."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}|\d+(?:\.\d+)?", text)]
    if not words:
        return inv_df.head(0)
    low = inv_df["Item"].str.lower()
    score = pd.Series(0, index=inv_df.index)
    for w in words:
        score = score + low.str.contains(re.escape(w), regex=True).astype(int)
    top = score.sort_values(ascending=False)
    top = top[top > 0].head(limit)
    return inv_df.loc[top.index]


SKU_PROMPT = """You pick the correct SKU for an invoice line from a shortlist of
inventory items. Match on product family, size/capacity, colour and finish.

Return ONLY valid JSON, no markdown fence:
{"sku": "", "item": "", "confidence": "high" | "medium" | "low", "reason": ""}

If no shortlist row is genuinely the same product, return sku "" with
confidence "low" and say what is missing. Never invent a SKU code.
"""


def find_sku(client, model, inv_df, description):
    cands = candidate_skus(inv_df, description)
    if cands.empty:
        return {"sku": "", "item": "", "confidence": "low",
                "reason": "no keyword match in inventory sheet"}
    payload = {
        "INVOICE_LINE": description,
        "CANDIDATES": cands.to_dict(orient="records"),
    }
    blocks = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    result, _ = call_claude_json(client, model, blocks, SKU_PROMPT, max_tokens=800)
    return result


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Invoice vs PR Checker")
st.caption("Upload the supplier bill and the Purchase Receive. Get a line-by-line check.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Anthropic API key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get one at console.anthropic.com. Set ANTHROPIC_API_KEY to skip this.",
    )
    model = MODELS[st.selectbox("Model", list(MODELS.keys()))]
    st.divider()
    inv_sheet = st.file_uploader("Inventory Summary (.xlsx)", type=["xlsx"])
    st.caption("Used to look up SKU codes for lines missing from the PR.")

col1, col2 = st.columns(2)
with col1:
    invoice_files = st.file_uploader(
        "Supplier invoice — photos or PDF (all pages)",
        type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True,
    )
with col2:
    pr_file = st.file_uploader("Purchase Receive (PDF)", type=["pdf"])

# --- parse the PR as soon as it is uploaded -------------------------------
pr_df, pr_meta = None, {}
if pr_file:
    pr_bytes = pr_file.getvalue()
    pr_df, pr_meta = parse_pr_pdf(pr_bytes)
    st.subheader("PR lines read from the PDF")
    st.caption(
        f"{pr_meta.get('pr','')} · PO {pr_meta.get('po','')} · "
        f"{pr_meta.get('vendor','')} · {len(pr_df)} lines · "
        f"total qty {pr_df['PR qty'].sum():,.0f}" if pr_df is not None and not pr_df.empty else ""
    )
    pr_df = st.data_editor(pr_df, use_container_width=True, num_rows="dynamic",
                           key="pr_editor")
    st.caption("Check these against the PDF and fix anything wrong before comparing.")

run = st.button("Check invoice against PR", type="primary",
                disabled=not (invoice_files and pr_file and api_key))

if run:
    client = Anthropic(api_key=api_key)

    # 1. invoice images
    images = []
    for f in invoice_files:
        data = f.getvalue()
        if f.name.lower().endswith(".pdf"):
            for png in pdf_pages_to_png(data):
                images.append((png, "image/png"))
        else:
            mt = "image/png" if f.name.lower().endswith(".png") else "image/jpeg"
            images.append((data, mt))

    if len(images) > 8:
        st.warning(f"{len(images)} pages — only the first 8 will be read.")
        images = images[:8]

    with st.spinner(f"Reading the invoice ({len(images)} page(s))…"):
        invoice_json, usage1 = read_invoice(client, model, images)

    st.subheader("Invoice")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Invoice no.", invoice_json.get("invoice_no", "—"))
    meta_cols[1].metric("PO no.", invoice_json.get("po_number", "—"))
    meta_cols[2].metric("Lines", len(invoice_json.get("lines", [])))
    meta_cols[3].metric("Total", invoice_json.get("invoice_total", "—"))

    inv_lines = pd.DataFrame(invoice_json.get("lines", []))
    with st.expander("Invoice lines as read"):
        st.dataframe(inv_lines, use_container_width=True)

    # 2. match
    with st.spinner("Matching line by line…"):
        matched, usage2 = match_lines(client, model, invoice_json, pr_df)

    rows = pd.DataFrame(matched.get("rows", []))
    if rows.empty:
        st.error("No rows came back from the match step.")
        st.stop()

    # 3. SKU lookup for lines with no PR SKU
    inv_master = load_inventory(inv_sheet.getvalue()) if inv_sheet else None
    if inv_master is not None:
        need = rows[(rows["pr_sku"].fillna("") == "") &
                    (rows["status"] != "not_in_invoice")]
        if not need.empty:
            with st.spinner(f"Looking up SKUs for {len(need)} unmatched line(s)…"):
                for idx, r in need.iterrows():
                    res = find_sku(client, model, inv_master, r["invoice_item"])
                    if res.get("sku"):
                        rows.at[idx, "pr_sku"] = res["sku"]
                        tag = f"SKU from inventory ({res.get('confidence','')}): {res.get('item','')}"
                        rows.at[idx, "note"] = (
                            f"{r.get('note','')} — {tag}".strip(" —")
                        )
                    else:
                        rows.at[idx, "note"] = (
                            f"{r.get('note','')} — no SKU in inventory sheet"
                        ).strip(" —")

    # 4. results
    display = rows.rename(columns={
        "inv": "Inv", "invoice_item": "Invoice item", "pr_sku": "PR SKU",
        "inv_qty": "Inv qty", "pr_qty": "PR qty", "status": "Status",
        "difference": "Diff", "note": "Note",
    })
    order = ["Inv", "Invoice item", "PR SKU", "Inv qty", "PR qty", "Status", "Diff", "Note"]
    display = display[[c for c in order if c in display.columns]]

    inv_total = pd.to_numeric(display["Inv qty"], errors="coerce").fillna(0).sum()
    pr_total = pd.to_numeric(display["PR qty"], errors="coerce").fillna(0).sum()
    problems = display[display["Status"] != "match"]

    m = st.columns(4)
    m[0].metric("Invoice qty", f"{inv_total:,.0f}")
    m[1].metric("PR qty", f"{pr_total:,.0f}")
    m[2].metric("Difference", f"{inv_total - pr_total:+,.0f}")
    m[3].metric("Lines to action", len(problems))

    def highlight(row):
        colour = {"match": "", "short": "background-color:#ffe0e0",
                  "excess": "background-color:#fff3cd",
                  "not_in_pr": "background-color:#ffe0e0",
                  "not_in_invoice": "background-color:#e0e8ff"}.get(row["Status"], "")
        return [colour] * len(row)

    st.subheader("Line-by-line check")
    st.dataframe(display.style.apply(highlight, axis=1),
                 use_container_width=True, height=560)

    if matched.get("summary"):
        st.info(matched["summary"])

    if not problems.empty:
        st.subheader("Needs action")
        st.dataframe(problems, use_container_width=True)

    st.download_button(
        "Download as CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name=f"check_{pr_meta.get('pr','result')}.csv",
        mime="text/csv",
    )

    tok = (usage1.input_tokens + usage1.output_tokens +
           usage2.input_tokens + usage2.output_tokens)
    st.caption(f"Model: {model} · about {tok:,} tokens used for this check.")

elif not api_key:
    st.info("Enter your Anthropic API key in the sidebar to start.")
