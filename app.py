import os
import json
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai

# -------------------- Config -------------------- #
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
    st.error("⚠️ GEMINI_API_KEY not found in environment variables.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY.strip())
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

MAX_CHARS_PER_CHUNK = 8000

# -------------------- Ministry Color Palette -------------------- #
COLOR_NAVY = "#132633"    # الرسالة
COLOR_MAROON = "#731e47"  # الرؤية
COLOR_ORANGE = "#f7931e"  # underline
COLOR_TEXT = "#0b2233"    # dark text
COLOR_WHITE = "#ffffff"   # background


# -------------------- Helpers -------------------- #
def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Split long text into character-based chunks."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        split_pos = text.rfind(".", start, end)
        if split_pos == -1 or split_pos <= start + int(max_chars * 0.5):
            split_pos = end
        chunks.append(text[start:split_pos].strip())
        start = split_pos
    return chunks


def clean_json_output(raw: str) -> str:
    """Remove markdown fences / extra text and keep outer JSON object."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = "".join(parts[1:-1]).strip()
        else:
            raw = raw.replace("```", "").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    first_brace, last_brace = raw.find("{"), raw.rfind("}")
    if first_brace != -1 and last_brace != -1:
        raw = raw[first_brace:last_brace + 1]
    return raw


def try_parse_json(raw: str) -> Dict[str, Any]:
    """Try parsing JSON, with a simple fallback clean step."""
    cleaned = clean_json_output(raw)
    try:
        return json.loads(cleaned)
    except Exception as e1:
        cleaned2 = "".join(ch for ch in cleaned if ord(ch) >= 32 or ch in "\n\r\t")
        try:
            return json.loads(cleaned2)
        except Exception as e2:
            raise ValueError(f"JSON parsing failed: {e1}; fallback: {e2}")


def analyze_chunk(text: str, idx: int, total: int) -> Dict[str, Any]:
    """Call Gemini on one chunk and return parsed JSON (or error info)."""
    prompt = f"""
You are a content moderation assistant for Arabic and English books.
Return ONLY JSON as per the schema below. No text outside JSON.

{{
  "harmful": true or false,
  "issues": [
    {{
      "phrase": "...",
      "category": "...",
      "severity": "Low | Medium | High",
      "language": "ar | en | mixed",
      "start_index": null,
      "end_index": null
    }}
  ],
  "summary": {{
    "categories": [],
    "total_issues": 0,
    "confidence": 90
  }}
}}

If no harmful content: same format with harmful=false and empty arrays.
Text to analyze:
{text}
"""
    try:
        res = model.generate_content(prompt)
        raw = res.text or ""
        parsed = try_parse_json(raw)
        parsed["_raw"] = raw
        parsed["_chunk_index"] = idx
        return parsed
    except Exception as e:
        return {
            "harmful": False,
            "issues": [],
            "summary": {"categories": [], "total_issues": 0, "confidence": 0},
            "_error": f"Chunk {idx+1} parse error: {str(e)}",
            "_raw": res.text if 'res' in locals() and hasattr(res, "text") else "",
            "_chunk_index": idx,
        }


def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze full text: chunking + per-chunk calls + aggregation."""
    chunks = chunk_text(text)
    all_results, all_issues, cats = [], [], set()
    total_issues, any_harmful, errors, max_conf = 0, False, [], 0

    for i, ch in enumerate(chunks):
        r = analyze_chunk(ch, i, len(chunks))
        all_results.append(r)

        if "_error" in r:
            errors.append(r["_error"])
        if r.get("harmful"):
            any_harmful = True

        s = r.get("summary", {}) or {}
        total_issues += s.get("total_issues", 0)
        cats |= set(s.get("categories", []))
        max_conf = max(max_conf, s.get("confidence", 0))

        for issue in r.get("issues", []) or []:
            issue["chunk_index"] = i + 1  # still kept internally
            all_issues.append(issue)

    return {
        "harmful": any_harmful,
        "issues": all_issues,
        "summary": {
            "categories": sorted(list(cats)),
            "total_issues": total_issues,
            "confidence": max_conf,
            "total_chunks": len(chunks),
        },
        "chunk_results": all_results,
        "errors": errors,
    }


# -------------------- Streamlit UI -------------------- #

def main():
    st.set_page_config(
        page_title="Ministry of Culture | Content Moderation System",
        page_icon="moc.png",
        layout="wide",
    )

    # --- Custom CSS (colors + buttons) ---
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {COLOR_WHITE};
            color: {COLOR_TEXT};
        }}
        h1, h2, h3 {{
            color: {COLOR_NAVY};
        }}
        .stButton>button {{
            background-color: {COLOR_MAROON};
            color: white;
            border-radius: 8px;
            font-weight: 600;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: {COLOR_ORANGE};
            color: {COLOR_NAVY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Header Section with logo + title ---
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.image("moc.png", width=100)  # make sure moc.png is in same folder
    with col_title:
        st.markdown(
            f"""
            <h1 style="color:{COLOR_MAROON}; margin-bottom:0;">
                Ministry of Culture | Content Moderation System
            </h1>
            <hr style="border-top: 3px solid {COLOR_ORANGE}; width: 120px; margin: 0;">
            """,
            unsafe_allow_html=True,
        )

    st.write("Analyze Arabic and English book text for harmful or sensitive content.")
    st.markdown("---")

    # --- Input Area (single full-width column) ---
    st.subheader("1. Input Text or Upload File")

    text_input = st.text_area(
        "Enter or paste text:",
        height=220,
        placeholder="Paste a page, chapter, or sample from the book (Arabic, English, or both)...",
    )

    uploaded_file = st.file_uploader(
        "Or upload a file (.txt or .pdf)",
        type=["txt", "pdf"],
    )

    # --- File Upload Logic ---
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            pages = [p.extract_text() or "" for p in reader.pages]
            text_input = "\n\n".join(pages)
            st.success(f"✅ Extracted text from {len(pages)} PDF pages.")
        else:
            text_input = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
            st.success("✅ Text file loaded successfully.")

    if text_input:
        st.caption(f"Text length: {len(text_input):,} characters")

    st.markdown("---")
    st.subheader("2. Run Analysis")

    # --- Run Analysis ---
    if st.button("🔍 Analyze Content"):
        if not text_input.strip():
            st.warning("Please enter or upload text first.")
            st.stop()

        with st.spinner("Analyzing content..."):
            result = analyze_text(text_input)

        s = result["summary"]
        harmful = result["harmful"]

        st.subheader("3. Summary")

        # Branded harmful / safe banner
        if harmful:
            st.markdown(
                f"""
                <div style="background-color:{COLOR_MAROON};
                            color:white;
                            padding:10px;
                            border-radius:6px;
                            font-weight:600;">
                    ⚠️ Harmful or sensitive content detected.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:{COLOR_NAVY};
                            color:white;
                            padding:10px;
                            border-radius:6px;
                            font-weight:600;">
                    ✅ No harmful content detected.
                </div>
                """,
                unsafe_allow_html=True,
            )

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Harmful", "Yes" if harmful else "No")
        colB.metric("Issues", s["total_issues"])
        colC.metric("Confidence", f"{s['confidence']}%")
        colD.metric("Chunks", s["total_chunks"])

        st.markdown(
            "**Categories:** "
            + (", ".join(s["categories"]) if s["categories"] else "None")
        )

        # --------- Detailed Issues with colored severity ---------
        st.subheader("4. Detailed Issues")
        if not result["issues"]:
            st.info("No specific harmful phrases were listed by the model.")
        else:
            for i, issue in enumerate(result["issues"], 1):
                expander_title = f"Issue {i}"
                with st.expander(expander_title):
                    phrase = issue.get("phrase", "N/A")
                    language = issue.get("language", "N/A")
                    category = issue.get("category", "N/A")
                    severity = issue.get("severity", "N/A") or "N/A"
                    severity = severity.strip().capitalize()

                    color_map = {
                        "High": "#d32f2f",   # Red
                        "Medium": "#f7931e", # Amber
                        "Low": "#388e3c",    # Green
                    }
                    color = color_map.get(severity, COLOR_TEXT)
                    border_color = color_map.get(severity, "#cccccc")

                    st.markdown(
                        f"""
                        <div style="border-left:6px solid {border_color};
                                    padding-left:10px;">
                            <p><strong>Phrase:</strong> `{phrase}`</p>
                            <p><strong>Language:</strong> {language}</p>
                            <p><strong>Category:</strong> {category}</p>
                            <p><strong>Severity level:</strong>
                                <span style="color:{color}; font-weight:600;">
                                    {severity}
                                </span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        # --------------------------------------------------------

        if result["errors"]:
            st.warning("Some chunks had parsing issues.")
            with st.expander("⚠️ Parsing Errors"):
                for e in result["errors"]:
                    st.text(e)

        with st.expander("🔧 Raw Model Outputs"):
            for c in result["chunk_results"]:
                st.code(c.get("_raw", ""), language="json")

    st.markdown("---")
    st.caption(
        "Prototype tool for research and internal evaluation – "
        "not a standalone safety layer for production systems."
    )


if __name__ == "__main__":
    main()
