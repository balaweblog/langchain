from langchain_core.prompts import ChatPromptTemplate

import base64
import os
import streamlit as st 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

st.title("Image findings with Multimodal LLM")

llm = ChatOllama(
    model="llava"
)

import json
import difflib

st.write("Upload a single document (image) and provide the Name and Phone Number to verify they appear in the document with the required labels.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])
name_input = st.text_input('Name (as shown on your income tax return)')
phone_input = st.text_input('Phone Number (List account number(s) here (optional))')
verify_button = st.button("Verify document")

if uploaded_file is not None and verify_button:
    if not name_input or not phone_input:
        st.warning("Please provide both Name and Phone Number to verify.")
    else:
        # Basic size check to avoid extremely large uploads
        try:
            size_bytes = uploaded_file.size
        except Exception:
            size_bytes = None

        if size_bytes and size_bytes > 10 * 1024 * 1024:  # 10 MB
            st.error("File too large. Please upload a file smaller than 10 MB or extract text locally before verification.")
        else:
            # PDF handling: extract text locally instead of sending raw PDF to the model
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                pdf_bytes = uploaded_file.read()
                extracted_text = ""
                try:
                    from PyPDF2 import PdfReader
                    import io

                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    pages = []
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    extracted_text = "\n".join(pages)
                except Exception:
                    extracted_text = ""

                if extracted_text.strip():
                    # local text verification
                    import re

                    def find_label_and_value(text, label, user_value):
                        """Return (label_found, value_found, matches, nearest_distance).

                        Uses:
                        - fuzzy name matching (difflib) and digit-only phone matching
                        - searches in a 400-char window before/after each label occurrence
                        - falls back to scanning the document for nearest occurrence and checks proximity
                        """
                        found_label = False
                        value_found = None
                        matches = False
                        nearest_distance = None

                        label_positions = [m.start() for m in re.finditer(re.escape(label), text, flags=re.IGNORECASE)]

                        def normalize_name(s: str) -> str:
                            return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()

                        def digits_only(s: str) -> str:
                            return re.sub(r"\D", "", s)

                        user_digits = digits_only(user_value)
                        user_norm = normalize_name(user_value)

                        if label_positions:
                            found_label = True

                            for lp in label_positions:
                                start = max(0, lp - 400)
                                end = min(len(text), lp + 400)
                                window = text[start:end]

                                # quick snippet immediately after label
                                m = re.search(re.escape(label) + r"\s*[:\-]?\s*(.{0,200})", window, flags=re.IGNORECASE | re.DOTALL)
                                if m and not value_found:
                                    snippet = m.group(1).strip()
                                    value_found = snippet if snippet else None

                                # phone matching by digits
                                if user_digits and len(user_digits) >= 4:
                                    for pm in re.finditer(r"[\d\-\(\)\.\s\+]{4,}", window):
                                        pm_text = pm.group(0)
                                        if user_digits in digits_only(pm_text):
                                            matches = True
                                            value_found = pm_text.strip()
                                            pos = start + pm.start()
                                            dist = min(abs(pos - lp), abs((start + pm.end()) - lp))
                                            if nearest_distance is None or dist < nearest_distance:
                                                nearest_distance = dist
                                            break
                                    if matches:
                                        break

                                # name matching per-line with fuzzy ratio
                                if user_norm:
                                    for line in re.split(r"\n+", window):
                                        ln = line.strip()
                                        if not ln:
                                            continue
                                        ln_norm = normalize_name(ln)
                                        if not ln_norm:
                                            continue
                                        if user_norm in ln_norm:
                                            matches = True
                                            value_found = ln
                                            pos = start + window.find(line)
                                            dist = abs(pos - lp)
                                            if nearest_distance is None or dist < nearest_distance:
                                                nearest_distance = dist
                                            break
                                        if difflib.SequenceMatcher(None, user_norm, ln_norm).ratio() >= 0.7:
                                            matches = True
                                            value_found = ln
                                            pos = start + window.find(line)
                                            dist = abs(pos - lp)
                                            if nearest_distance is None or dist < nearest_distance:
                                                nearest_distance = dist
                                            break
                                    if matches:
                                        break

                            # fallback: scan entire document for nearest occurrence near any label
                            if not matches:
                                if user_digits and len(user_digits) >= 4:
                                    for m in re.finditer(r"[\d\-\(\)\.\s\+]{4,}", text):
                                        if user_digits in digits_only(m.group(0)):
                                            pos = m.start()
                                            min_dist = min(abs(pos - lp) for lp in label_positions)
                                            if min_dist <= 400:
                                                matches = True
                                                value_found = m.group(0).strip()
                                                nearest_distance = min_dist
                                                break
                                if not matches and user_norm:
                                    for m in re.finditer(r"([^
                                        line = m.group(1).strip()
                                        if not line:
                                            continue
                                        line_norm = normalize_name(line)
                                        if not line_norm:
                                            continue
                                        if user_norm in line_norm or difflib.SequenceMatcher(None, user_norm, line_norm).ratio() >= 0.7:
                                            pos = m.start()
                                            min_dist = min(abs(pos - lp) for lp in label_positions)
                                            if min_dist <= 400:
                                                matches = True
                                                value_found = line
                                                nearest_distance = min_dist
                                                break

                        return found_label, value_found, matches, nearest_distance

                    LABEL_NAME = "Name (as shown on your income tax return)"
                    LABEL_PHONE = "List account number(s) here (optional)"

                    name_label_present, name_value_found, name_matches, name_distance = find_label_and_value(extracted_text, LABEL_NAME, name_input)
                    phone_label_present, phone_value_found, phone_matches, phone_distance = find_label_and_value(extracted_text, LABEL_PHONE, phone_input)

                    overall_ok = all([name_label_present, name_matches, phone_label_present, phone_matches])
                    st.json({
                        "name_label_present": bool(name_label_present),
                        "name_value_found": name_value_found,
                        "name_matches": bool(name_matches),
                        "name_distance": name_distance,
                        "phone_label_present": bool(phone_label_present),
                        "phone_value_found": phone_value_found,
                        "phone_matches": bool(phone_matches),
                        "phone_distance": phone_distance,
                        "overall_ok": bool(overall_ok),
                    })
                else:
                    st.error("Could not extract text from PDF. If your PDF is scanned (images), enable OCR support by installing 'pdf2image' and 'pytesseract' with system 'poppler' and 'tesseract' binaries, or upload the PDF as images.")
                    st.markdown("**Quick tips:**\n- Install poppler and tesseract on macOS: `brew install poppler tesseract`\n- Then pip install `pdf2image pytesseract` and re-run to enable OCR-based verification.")
            else:
                # Image path — use the multimodal LLM but with robust error handling
                image_bytes = uploaded_file.read()
                image_data = base64.b64encode(image_bytes).decode()

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful assistant that can read images and locate text in them. "
                                   "Check whether the image contains the exact labels (case-insensitive substring match) "
                                   "'Name (as shown on your income tax return)' and "
                                   "'List account number(s) here (optional)' (ensure parentheses and wording are present). "
                                   "For each label, check whether the user's provided value appears next to or below it. "
                                   "Return a JSON object with keys: name_label_present (true/false), "
                                   "name_value_found (string|null), name_matches (true/false), "
                                   "phone_label_present, phone_value_found, phone_matches, and overall_ok (true/false). "
                                   "If you find text, include the extracted text. Be concise."),
                        (
                            "human",
                            [
                                {"type": "text", "text": f"Name (as shown on your income tax return) {name_input}\nList account number(s) here (optional) {phone_input}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}",
                                        "detail": "low",
                                    },
                                },
                            ],
                        ),
                    ]
                )

                chain = prompt | llm
                try:
                    response = chain.invoke({})
                    resp_text = str(response)

                    try:
                        parsed = json.loads(resp_text)
                        st.json(parsed)
                    except Exception:
                        st.write("Model response (raw):")
                        st.write(resp_text)
                        st.info("If the response above is not JSON, please re-run or adjust the prompt to return JSON.")
                except Exception as e:
                    err_str = str(e)
                    st.error("Model invocation failed: " + err_str)
                    if "model runner has unexpectedly stopped" in err_str or "ResponseError" in err_str:
                        st.warning("This error often means the Ollama model runner stopped (resource limits or internal error). Check the Ollama server logs for details. Consider reducing the file size or verifying the Ollama model is running with sufficient resources.")
                    st.stop()