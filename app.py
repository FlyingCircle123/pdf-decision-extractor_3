# =========================
# IMPORTS
# =========================
import streamlit as st
import PyPDF2
from openai import OpenAI
import os
import time
import json
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import hashlib
import csv
from io import StringIO

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="PDF Decision Extractor", layout="wide")
CHUNK_SIZE = 1000
OVERLAP_SIZE = 200
MODEL = "gpt-3.5-turbo"
MAX_FILE_SIZE_MB = 50  # File size warning

# =========================
# SESSION STATE INIT
# =========================
if "chunk_results" not in st.session_state:
    st.session_state.chunk_results = []
if "text_cache" not in st.session_state:
    st.session_state.text_cache = {}
if "last_pdf_hash" not in st.session_state:
    st.session_state.last_pdf_hash = None

# =========================
# HELPERS
# =========================
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def file_size_mb(file_bytes):
    return len(file_bytes) / (1024*1024)

# =========================
# PDF PROCESSING
# =========================
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"[PAGE {i+1}]\n{page_text}\n"
        return text
    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {str(e)}")
        return ""

def extract_text_with_ocr(pdf_file):
    text = ""
    try:
        pdf_file.seek(0)
        images = convert_from_bytes(pdf_file.read(), dpi=200)
        total_pages = len(images)
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, page in enumerate(images):
            status_text.text(f"OCR page {i+1}/{total_pages}...")
            page_text = pytesseract.image_to_string(page)
            if page_text:
                text += f"[PAGE {i+1}]\n{page_text}\n"
            progress_bar.progress((i+1)/total_pages)
        status_text.text("OCR complete!")
        return text
    except Exception as e:
        st.error(f"OCR failed: {str(e)}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    chunks = []
    for i in range(0, len(text), chunk_size-overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# =========================
# AI ENGINE
# =========================
def build_prompt(chunk):
    return f"""
    Extract all key decisions, action items, and important points from this text.
    
    Return ONLY valid JSON with this exact structure:
    {{
        "decisions": ["specific decision 1", "specific decision 2"],
        "action_items": ["action item 1", "action item 2"],
        "key_points": ["key point 1", "key point 2"]
    }}
    
    Rules:
    - If a category has no items, use empty list []
    - Be specific and concise
    - Extract directly from the text, don't invent
    - Return ONLY the JSON, no other text
    
    Text: {chunk}
    """

def call_ai(prompt, client, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json","").replace("```","").strip()
            elif content.startswith("```"):
                content = content.replace("```","").strip()
            elif content.startswith('"""') and content.endswith('"""'):
                content = content.replace('"""','').strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "raw_content": content,
                "note": "AI didn't return valid JSON",
                "decisions": [],
                "action_items": [],
                "key_points":[f"Raw extraction (not structured): {content[:200]}..."]
            }
        except Exception as e:
            if attempt < retries-1:
                time.sleep(delay * (2 ** attempt))
                continue
            return {
                "error": str(e),
                "decisions": [],
                "action_items": [],
                "key_points":[f"Error processing chunk: {str(e)}"]
            }

def merge_results(results):
    merged = {"decisions":[],"action_items":[],"key_points":[]}
    for r in results:
        if isinstance(r, dict):
            for k in merged.keys():
                if k in r and isinstance(r[k], list):
                    merged[k].extend(r[k])
            if "raw_content" in r and r["raw_content"]:
                merged["key_points"].append(f"[Raw chunk]: {r['raw_content'][:100]}...")
    for k in merged.keys():
        seen=set()
        merged[k]=[x for x in merged[k] if not (x in seen or seen.add(x))]
    return merged

def process_document(chunks, client, force_reprocess=False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Cache read
    if not force_reprocess and st.session_state.last_pdf_hash in st.session_state.text_cache:
        status_text.text("Loading results from cache...")
        return st.session_state.text_cache[st.session_state.last_pdf_hash]
    
    chunk_results = []
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
        prompt = build_prompt(chunk)
        result = call_ai(prompt, client)
        chunk_results.append(result)
        st.session_state.chunk_results.append(result)
        progress_bar.progress((i+1)/len(chunks))
    
    final_result = merge_results(chunk_results)
    
    # Cache write
    if st.session_state.last_pdf_hash:
        st.session_state.text_cache[st.session_state.last_pdf_hash] = final_result
    
    status_text.text("Merging results...")
    return final_result

# =========================
# UI
# =========================
def render_output(result):
    st.markdown("## 📋 Extracted Decisions")
    
    if "error" in result and result["error"]:
        st.error(f"Error: {result['error']}")
        return
    
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Decisions")
        if result.get("decisions"):
            for d in result["decisions"]:
                st.markdown(f"- {d}")
        else:
            st.markdown("*No decisions found*")
        st.markdown("### ⚡ Action Items")
        if result.get("action_items"):
            for a in result["action_items"]:
                st.markdown(f"- {a}")
        else:
            st.markdown("*No action items found*")
    with col2:
        st.markdown("### 💡 Key Points")
        if result.get("key_points"):
            for k in result["key_points"]:
                st.markdown(f"- {k}")
        else:
            st.markdown("*No key points found*")
    
    # JSON download
    st.download_button(
        label="📥 Download JSON",
        data=json.dumps(result, indent=2),
        file_name="extracted_decisions.json",
        mime="application/json"
    )
    
    # CSV download
    output_csv = StringIO()
    writer = csv.writer(output_csv)
    writer.writerow(["Category","Item"])
    for cat in ["decisions","action_items","key_points"]:
        for item in result.get(cat,[]):
            writer.writerow([cat,item])
    st.download_button(
        label="📥 Download CSV",
        data=output_csv.getvalue(),
        file_name="extracted_decisions.csv",
        mime="text/csv"
    )

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY","")

def main():
    st.title("📄 PDF Decision Extractor (VIBE MODE)")
    st.markdown("Upload a PDF to extract key decisions, actions, and insights.")
    
    api_key = get_api_key()
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password")
        else:
            st.success("✅ API key loaded from secrets/env")
        st.markdown("---")
        st.markdown("**Tips:** scanned PDFs use OCR; large files may take time")
    
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        uploaded_file.seek(0,os.SEEK_END)
        size_mb = file_size_mb(uploaded_file.read())
        uploaded_file.seek(0)
        if size_mb > MAX_FILE_SIZE_MB:
            st.warning(f"⚠️ File is {size_mb:.1f}MB — might be slow or fail")
        
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        pdf_hash = get_file_hash(file_bytes)
        st.session_state.last_pdf_hash = pdf_hash
        
        reprocess = st.checkbox("🔄 Re-process (ignore cache)")
        if st.button("🚀 Extract Decisions"):
            if not api_key:
                st.warning("⚠️ Please enter your OpenAI API key")
            else:
                client = OpenAI(api_key=api_key)
                
                text = extract_text_from_pdf(uploaded_file)
                if not text.strip():
                    st.warning("No text found — trying OCR fallback...")
                    text = extract_text_with_ocr(uploaded_file)
                if not text.strip():
                    st.error("❌ Could not extract text from PDF")
                    st.stop()
                
                st.info(f"📊 Extracted {len(text)} chars, {len(text.split())} words")
                
                chunks = chunk_text(text)
                st.info(f"📦 Split into {len(chunks)} chunks")
                
                result = process_document(chunks, client, force_reprocess=reprocess)
                render_output(result)
                st.success("✅ Extraction complete!")
    else:
        st.info("👆 Upload a PDF to get started")

if __name__=="__main__":
    main()
