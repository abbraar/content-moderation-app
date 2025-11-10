import os
import json
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai

# -------------------- Config & Setup -------------------- #

# Load environment variables from .env file in the same directory as the script
load_dotenv()

# Get the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
    st.error("⚠️ GEMINI_API_KEY not found in environment variables. Please set it in your deployment environment.")
    st.stop()

# Configure Gemini model
genai.configure(api_key=GEMINI_API_KEY.strip())
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Maximum characters per chunk to send to the model
MAX_CHARS_PER_CHUNK = 8000


# -------------------- Helpers -------------------- #

def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Split long text into character-based chunks."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to cut at a sentence boundary if possible
        split_pos = text.rfind(".", start, end)
        if split_pos == -1 or split_pos <= start + int(max_chars * 0.5):
            split_pos = end
        chunks.append(text[start:split_pos].strip())
        start = split_pos
    return chunks


def clean_json_output(raw: str) -> str:
    """
    Clean common LLM wrappers around JSON, like ```json ... ``` fences.
    Returns a string that should be valid JSON or close to it.
    """
    raw = raw.strip()

    # Remove Markdown fences like ```json ... ```
    if raw.startswith("```"):
        # remove leading ``` and trailing ``` if present
        # (do it more carefully than just strip("`"))
        parts = raw.split("```")
        # Everything between first and last ``` is the content
        if len(parts) >= 3:
            raw = "".join(parts[1:-1]).strip()
        else:
            raw = raw.replace("```", "").strip()

        # Remove possible 'json' language identifier at the start
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # Sometimes the model adds explanations before/after JSON
    # Try to keep only the outermost {...} or [...]
    first_brace = raw.find("{")
    first_bracket = raw.find("[")
    if first_brace == -1 and first_bracket == -1:
        return raw

    # Choose the earliest opening of { or [
    if first_brace == -1 or (first_bracket != -1 and first_bracket < first_brace):
        # JSON array
        first = first_bracket
        last = raw.rfind("]")
    else:
        # JSON object
        first = first_brace
        last = raw.rfind("}")

    if first != -1 and last != -1 and last > first:
        raw = raw[first:last + 1].strip()

    return raw


def try_parse_json(raw: str) -> Dict[str, Any]:
    """
    Try to parse JSON. If it fails, try a couple of simple fixes.
    Raise the original error if still not valid.
    """
    cleaned = clean_json_output(raw)

    # First attempt: direct load
    try:
        return json.loads(cleaned)
    except Exception as e1:
        # Simple fallback: remove control characters that sometimes break JSON
        cleaned2 = "".join(ch for ch in cleaned if ord(ch) >= 32 or ch in "\n\r\t")
        try:
            return json.loads(cleaned2)
        except Exception as e2:
            # Re-raise with both error messages for debugging
            raise ValueError(
                f"Primary JSON error: {e1}; "
                f"Secondary JSON error after cleaning: {e2}; "
                f"Cleaned text (first 500 chars): {cleaned2[:500]!r}"
            )


def analyze_chunk(text: str, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
    """
    Analyze a single chunk of text and return parsed JSON result.
    The schema is defined in the prompt.
    """
    system_prompt = """
You are a content moderation assistant for books in both Arabic and English.

Analyze the provided text for harmful, inappropriate, or offensive content.
Focus on content such as (but not limited to):

- Hate speech or insults towards individuals or groups
- Bullying / harassment
- Violence and physical harm
- Sexual content (including exploitation)
- Self-harm or suicide
- Extremism or terrorism
- Discrimination (religion, race, gender, etc.)
- Profanity or very offensive language
- Any content that may be harmful for children

The text can be in Arabic, English, or a mix of both.

You MUST respond **only** with raw JSON. No explanations, no Markdown, no code fences.
The JSON schema MUST be:

{
  "harmful": true or false,
  "issues": [
    {
      "phrase": "exact phrase that is harmful",
      "category": "short category name (e.g., hate_speech, violence, sexual, self_harm, profanity, extremism, discrimination, other)",
      "severity": "Low | Medium | High",
      "language": "ar | en | mixed",
      "start_index": optional integer (or null),
      "end_index": optional integer (or null)
    }
  ],
  "summary": {
    "categories": ["list", "of", "unique", "categories"],
    "total_issues": integer,
    "confidence": integer between 0 and 100
  }
}

If you do not find any harmful content, return exactly:

{
  "harmful": false,
  "issues": [],
  "summary": {
    "categories": [],
    "total_issues": 0,
    "confidence": 90
  }
}
""".strip()

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Chunk information: chunk_index={chunk_index + 1}, total_chunks={total_chunks}\n\n"
        f"Text to analyze:\n{text}"
    )

    try:
        response = model.generate_content(full_prompt)
        raw_text = response.text or ""
        parsed = try_parse_json(raw_text)
        parsed["_raw"] = raw_text  # keep raw for debugging
        parsed["_chunk_index"] = chunk_index
        return parsed
    except Exception as e:
        # Any parsing / API error will be captured here
        return {
            "harmful": False,
            "issues": [],
            "summary": {
                "categories": [],
                "total_issues": 0,
                "confidence": 0,
            },
            "_error": f"Error parsing model response for chunk {chunk_index + 1}: {str(e)}",
            "_raw": response.text if 'response' in locals() and hasattr(response, "text") else "",
            "_chunk_index": chunk_index,
        }


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze a full text (book or chapter). Handles chunking,
    calling the model for each chunk, and aggregating results.
    """
    chunks = chunk_text(text, MAX_CHARS_PER_CHUNK)
    total_chunks = len(chunks)

    all_chunk_results: List[Dict[str, Any]] = []
    global_issues: List[Dict[str, Any]] = []
    global_categories = set()
    max_confidence = 0
    total_issues = 0
    any_harmful = False
    errors: List[str] = []

    for idx, chunk in enumerate(chunks):
        result = analyze_chunk(chunk, idx, total_chunks)
        all_chunk_results.append(result)

        if "_error" in result:
            errors.append(result["_error"])

        if result.get("harmful"):
            any_harmful = True

        summary = result.get("summary", {}) or {}
        total_issues += summary.get("total_issues", 0) or 0
        for c in summary.get("categories", []) or []:
            global_categories.add(c)

        max_confidence = max(max_confidence, summary.get("confidence", 0) or 0)

        for issue in result.get("issues", []) or []:
            issue_copy = dict(issue)
            issue_copy["chunk_index"] = idx + 1  # 1-based
            global_issues.append(issue_copy)

    aggregated = {
        "harmful": any_harmful,
        "issues": global_issues,
        "summary": {
            "categories": sorted(list(global_categories)),
            "total_issues": total_issues,
            "confidence": max_confidence,
            "total_chunks": total_chunks,
        },
        "chunk_results": all_chunk_results,
        "errors": errors,
    }

    return aggregated


# -------------------- Streamlit UI -------------------- #

def main():
    st.set_page_config(
        page_title="Content Moderation System",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Book Content Moderation System")
    st.write(
        "Detect harmful or sensitive content in **Arabic and English** book text "
        "(content moderation prototype)."
    )

    st.markdown("### 1. Input Text or Upload File")

    # Left: text/file input, Right: info
    col_input, col_info = st.columns([2, 1])

    with col_input:
        text_input = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Paste a page, chapter, or sample from the book (Arabic, English, or both)...",
        )

        uploaded_file = st.file_uploader("Or upload a file (.txt or .pdf)", type=["txt", "pdf"])

    with col_info:
        st.markdown(
            f"- Max characters per chunk: **{MAX_CHARS_PER_CHUNK}**\n"
            "- Text is automatically split into chunks if it is long.\n"
            "- This is a prototype and not a final safety classifier."
        )

    # Handle file upload
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            try:
                pdf_reader = PdfReader(uploaded_file)
                extracted_text_parts = []

                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                    except Exception:
                        page_text = None

                    if page_text and page_text.strip():
                        extracted_text_parts.append(page_text)

                if not extracted_text_parts:
                    # No real text came out of the PDF
                    text_input = ""
                    if "uploaded_text" in st.session_state:
                        st.session_state.pop("uploaded_text", None)

                    st.error(
                        "No readable text could be extracted from this PDF.\n\n"
                        "This often happens when the book is a **scanned image** PDF, "
                        "or when the Arabic text is not stored as actual text."
                    )
                    st.info(
                        "Try another PDF or convert the file to a **searchable PDF** using OCR "
                        "(e.g., Tesseract, OlmOCR, or another OCR tool), then upload it again."
                    )
                    return

                text_input = "\n\n".join(extracted_text_parts).strip()
                st.session_state["uploaded_text"] = text_input
                st.success(f"✅ Extracted text from {len(pdf_reader.pages)} PDF pages.")
            except Exception as e:
                st.error(f"Error reading PDF file: {str(e)}")
                return
        else:
            # Plain text file
            try:
                text_input = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                text_input = text_input.strip()
                if not text_input:
                    st.error("The uploaded text file appears to be empty.")
                    return
                st.session_state["uploaded_text"] = text_input
                st.success("✅ Text file loaded successfully.")
            except Exception as e:
                st.error(f"Error reading text file: {str(e)}")
                return

    # Restore previous uploaded text if the text_area is empty
    if "uploaded_text" in st.session_state and not text_input:
        text_input = st.session_state["uploaded_text"]

    if text_input:
        st.markdown(f"**Text length:** {len(text_input):,} characters")

    st.markdown("### 2. Run Analysis")

    analyze_button = st.button("🔍 Analyze Content")

    if analyze_button:
        # Make sure we really have usable text
        if not text_input or not text_input.strip():
            st.warning(
                "No readable text is currently loaded.\n\n"
                "- If you uploaded a PDF, it may be a scanned/image-only file.\n"
                "- Try another file or preprocess it with OCR, then upload again."
            )
            st.stop()

        with st.spinner("Analyzing content (this may take a moment for long texts)..."):
            result = analyze_text(text_input)

        errors = result.get("errors", []) or []
        if errors:
            st.warning("Some chunks had parsing issues. See 'Parsing Errors' or 'Raw Chunk Outputs' below for details.")

        harmful = result.get("harmful", False)
        summary = result.get("summary", {}) or {}
        issues = result.get("issues", []) or []

        # -------------------- Summary -------------------- #
        st.markdown("### 3. Summary")

        if not harmful:
            st.success("✅ No harmful content detected in the analyzed text (within model limitations).")
        else:
            st.error("⚠️ Potential harmful or sensitive content detected.")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Harmful Detected", "Yes" if harmful else "No")
        with col_s2:
            st.metric("Total Issues", summary.get("total_issues", 0))
        with col_s3:
            st.metric("Confidence (max)", f"{summary.get('confidence', 0)}%")
        with col_s4:
            st.metric("Chunks Analyzed", summary.get("total_chunks", 1))

        st.markdown(
            "**Categories:** "
            + (", ".join(summary.get("categories", [])) if summary.get("categories") else "None")
        )

        # -------------------- Detailed Issues -------------------- #
        st.markdown("### 4. Detailed Issues")

        if not issues:
            st.info("No specific harmful phrases were listed by the model.")
        else:
            for i, issue in enumerate(issues, start=1):
                with st.expander(
                    f"Issue {i}: {issue.get('category', 'unknown')} "
                    f"({issue.get('severity', 'Unknown')})"
                ):
                    st.write(f"**Phrase:** `{issue.get('phrase', 'N/A')}`")
                    st.write(f"**Language:** {issue.get('language', 'N/A')}")
                    st.write(f"**Severity:** {issue.get('severity', 'N/A')}")
                    st.write(f"**Chunk Index:** {issue.get('chunk_index', 'N/A')}")

        # -------------------- Parsing Errors -------------------- #
        if errors:
            with st.expander("⚠️ Parsing Errors (debug)"):
                for e in errors:
                    st.write("-", e)

        # -------------------- Raw Outputs for Debugging -------------------- #
        with st.expander("🔧 Raw Chunk Outputs (for debugging / research)"):
            for chunk_result in result.get("chunk_results", []):
                idx = chunk_result.get("_chunk_index", 0) + 1
                harmful_chunk = chunk_result.get("harmful")
                st.markdown(f"**Chunk {idx} (harmful={harmful_chunk})**")
                if "_error" in chunk_result:
                    st.write(f"Error: {chunk_result['_error']}")
                st.code(chunk_result.get("_raw", ""), language="json")

    st.markdown("---")
    st.markdown(
        """
### How It Works

1. Paste text or upload a **.txt** / **.pdf** file (Arabic, English, or mixed).
2. The system splits long text into chunks and sends each chunk to Gemini.
3. Gemini returns a **JSON** report with harmful phrases, categories, severity, and language.
4. The app aggregates results and shows:
   - Whether harmful content was detected
   - Total number of issues
   - Categories and confidence
   - Detailed list of phrases

> ⚠️ This is a **research / prototyping** tool.  
> It should not be used as the only safety layer in production systems.
        """
    )


if __name__ == "__main__":
    main()
