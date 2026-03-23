from __future__ import annotations

import gc
import json
import time
import base64
import traceback
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
import requests
import streamlit as st


# =========================
# Configuration
# =========================

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"

for d in [DATA_DIR, UPLOAD_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_number": {"type": "integer"},
        "title": {"type": "string"},
        "main_points": {"type": "array", "items": {"type": "string"}},
        "important_numbers": {"type": "array", "items": {"type": "string"}},
        "visual_takeaways": {"type": "array", "items": {"type": "string"}},
        "jargon_to_explain": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "quote_lines": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "page_number",
        "title",
        "main_points",
        "important_numbers",
        "visual_takeaways",
        "jargon_to_explain",
        "caveats",
        "quote_lines"
    ]
}

DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "document_title": {"type": "string"},
        "executive_summary": {"type": "string"},
        "major_themes": {"type": "array", "items": {"type": "string"}},
        "key_statistics": {"type": "array", "items": {"type": "string"}},
        "policy_relevance": {"type": "array", "items": {"type": "string"}},
        "terms_to_define": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "document_title",
        "executive_summary",
        "major_themes",
        "key_statistics",
        "policy_relevance",
        "terms_to_define"
    ]
}

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "episode_title": {"type": "string"},
        "intro": {"type": "string"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["speaker", "text"]
            }
        },
        "closing": {"type": "string"}
    },
    "required": ["episode_title", "intro", "turns", "closing"]
}


# =========================
# Helpers
# =========================

def make_run_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{ts}"
    (run_dir / "pages").mkdir(parents=True, exist_ok=True)
    (run_dir / "facts").mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_path = target_dir / uploaded_file.name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def append_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def safe_delete(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def load_voices_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def warm_ollama_model(model: str, keep_alive: str = "30m"):
    payload = {
        "model": model,
        "messages": [],
        "keep_alive": keep_alive,
        "stream": False,
    }
    try:
        requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
    except Exception:
        pass


def ollama_chat_json(
    model: str,
    messages: list,
    schema: dict,
    temperature: float = 0.2,
    timeout: int = 1800,
    keep_alive: str = "30m",
):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature
        }
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data["message"]["content"]
    if isinstance(content, str):
        return json.loads(content)
    return content


# =========================
# PDF Processing
# =========================

def iter_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int = 120,
    max_pages: int | None = None,
    text_limit: int = 12000,
) -> Iterator[dict]:
    doc = fitz.open(pdf_path)

    try:
        total_pages = len(doc)
        limit = min(total_pages, max_pages) if max_pages else total_pages

        for i in range(limit):
            page = doc.load_page(i)
            text = page.get_text("text")[:text_limit]

            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img_path = pages_dir / f"{pdf_path.stem}_p{i+1}.png"
            img_path.write_bytes(pix.tobytes("png"))

            yield {
                "page": i + 1,
                "text": text,
                "image_path": str(img_path),
                "source_pdf": pdf_path.name,
            }

            del pix
            del page
            gc.collect()
    finally:
        doc.close()
        gc.collect()


def analyze_page(
    page_record: dict,
    model: str,
    use_vision: bool = True,
    delete_image_after: bool = False,
    timeout: int = 1800,
    keep_alive: str = "30m",
) -> dict:
    image_b64 = None

    try:
        prompt = f"""
You are reading one PDF page for a public-friendly economics podcast.

Use the extracted page text and, if provided, the page image.
Your job is to capture the content faithfully, especially charts, tables, infographics, titles, and key numbers.

Rules:
- Do not invent facts not visible in the text/image.
- If a chart or infographic is unclear, mention that in caveats.
- Explain the page in simple language.
- Prefer concrete numbers and findings.
- Keep the wording neutral and grounded in the page.
- This page comes from: {page_record['source_pdf']}

Page number: {page_record['page']}

Extracted text:
{page_record['text']}
""".strip()

        message = {
            "role": "user",
            "content": prompt,
        }

        if use_vision:
            image_bytes = Path(page_record["image_path"]).read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            message["images"] = [image_b64]
            del image_bytes

        result = ollama_chat_json(
            model=model,
            messages=[message],
            schema=PAGE_SCHEMA,
            temperature=0.1,
            timeout=timeout,
            keep_alive=keep_alive,
        )

        result["source_pdf"] = page_record["source_pdf"]
        result["analysis_mode"] = "vision" if use_vision else "text_only"
        result["status"] = "ok"
        return result

    finally:
        image_b64 = None
        gc.collect()
        if delete_image_after:
            safe_delete(Path(page_record["image_path"]))


def analyze_page_with_fallback(
    page_record: dict,
    model: str,
    preferred_use_vision: bool = True,
    delete_image_after: bool = False,
    first_timeout: int = 1200,
    second_timeout: int = 1800,
    keep_alive: str = "30m",
) -> dict:
    if preferred_use_vision:
        try:
            return analyze_page(
                page_record=page_record,
                model=model,
                use_vision=True,
                delete_image_after=delete_image_after,
                timeout=first_timeout,
                keep_alive=keep_alive,
            )
        except requests.exceptions.ReadTimeout:
            try:
                return analyze_page(
                    page_record=page_record,
                    model=model,
                    use_vision=True,
                    delete_image_after=delete_image_after,
                    timeout=second_timeout,
                    keep_alive=keep_alive,
                )
            except requests.exceptions.ReadTimeout:
                return analyze_page(
                    page_record=page_record,
                    model=model,
                    use_vision=False,
                    delete_image_after=delete_image_after,
                    timeout=second_timeout,
                    keep_alive=keep_alive,
                )
    else:
        return analyze_page(
            page_record=page_record,
            model=model,
            use_vision=False,
            delete_image_after=delete_image_after,
            timeout=second_timeout,
            keep_alive=keep_alive,
        )


def build_doc_brief(page_facts: list[dict], model: str, timeout: int = 1800, keep_alive: str = "30m") -> dict:
    prompt = f"""
You are combining page-level facts from one or more economics PDFs into a grounded document brief.

Audience:
General public.

Style:
Clear, concise, factual.

Rules:
- Base the brief only on the provided page facts.
- Do not invent missing details.
- If the evidence is partial, reflect that carefully.

Page facts:
{json.dumps(page_facts, ensure_ascii=False)}
""".strip()

    return ollama_chat_json(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        schema=DOC_SCHEMA,
        temperature=0.2,
        timeout=timeout,
        keep_alive=keep_alive,
    )


def build_script(doc_brief: dict, model: str, minutes: int = 5, timeout: int = 1800, keep_alive: str = "30m") -> dict:
    prompt = f"""
Write a podcast conversation between two people: Host A and Host B.

Audience:
General public.

Tone:
Friendly, warm, intelligent, simple, conversational.

Rules:
- Base the script ONLY on the provided document brief.
- Use easy language.
- Explain economics and technical terms simply.
- Keep turns fairly short.
- Sound natural, not robotic.
- No hallucinations.
- Include a strong intro hook.
- Include a short takeaway recap at the end.
- Aim for about {minutes} minutes total.

Document brief:
{json.dumps(doc_brief, ensure_ascii=False)}
""".strip()

    return ollama_chat_json(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        schema=SCRIPT_SCHEMA,
        temperature=0.5,
        timeout=timeout,
        keep_alive=keep_alive,
    )


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="BriefCast-AI", layout="wide")
st.title("BriefCast-AI")
st.caption("Local multimodal PDF-to-podcast app using Ollama and Kokoro")

if "run_dir" not in st.session_state:
    st.session_state.run_dir = None

if "facts_jsonl" not in st.session_state:
    st.session_state.facts_jsonl = None

if "script_json_path" not in st.session_state:
    st.session_state.script_json_path = None

if "doc_brief_path" not in st.session_state:
    st.session_state.doc_brief_path = None

if "error_jsonl" not in st.session_state:
    st.session_state.error_jsonl = None


with st.sidebar:
    st.header("Models")
    vlm_model = st.text_input("Vision model", value="qwen2.5vl:7b")
    writer_model = st.text_input("Writer model", value="qwen3.5:9b")

    st.header("Safe mode")
    use_vision = st.checkbox("Use page images", value=True)
    dpi = st.slider("Render DPI", min_value=80, max_value=220, value=120, step=10)
    max_pages = st.slider("Max pages per PDF", min_value=1, max_value=100, value=10, step=1)
    minutes = st.slider("Podcast length (minutes)", min_value=2, max_value=20, value=4, step=1)
    delete_images_after = st.checkbox("Delete page images after analysis", value=True)
    text_limit = st.slider("Max extracted text per page", min_value=2000, max_value=20000, value=12000, step=1000)

    st.header("Timeouts")
    first_timeout = st.slider("First page timeout (seconds)", min_value=120, max_value=1800, value=900, step=60)
    second_timeout = st.slider("Fallback timeout (seconds)", min_value=300, max_value=3600, value=1800, step=60)
    keep_alive = st.text_input("Ollama keep_alive", value="30m")

    st.header("Voices")
    voices_json_path = st.text_input("voices.json path", value="voices.json")
    lang_code = st.text_input("Kokoro lang code", value="a")
    speed = st.slider("Voice speed", min_value=0.7, max_value=1.3, value=1.0, step=0.05)
    export_mp3 = st.checkbox("Also export MP3", value=True)

    st.header("Maintenance")
    if st.button("Start new run"):
        st.session_state.run_dir = None
        st.session_state.facts_jsonl = None
        st.session_state.script_json_path = None
        st.session_state.doc_brief_path = None
        st.session_state.error_jsonl = None
        st.success("Session reset.")


uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2, col3 = st.columns(3)

with col1:
    process_btn = st.button(" FIRST: Process PDFs")

with col2:
    brief_btn = st.button("SECOND: Build document brief")

with col3:
    script_btn = st.button("THIRD: Write script")


if process_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        run_dir = make_run_dir()
        pages_dir = run_dir / "pages"
        facts_dir = run_dir / "facts"
        facts_jsonl = facts_dir / "page_facts.jsonl"
        error_jsonl = facts_dir / "page_errors.jsonl"

        st.session_state.run_dir = str(run_dir)
        st.session_state.facts_jsonl = str(facts_jsonl)
        st.session_state.error_jsonl = str(error_jsonl)

        saved_pdfs = []
        for up in uploaded_files:
            saved_pdfs.append(save_uploaded_file(up, UPLOAD_DIR))

        total_pages_estimate = len(saved_pdfs) * max_pages
        progress = st.progress(0)
        status = st.empty()
        processed_count = 0
        success_count = 0
        failure_count = 0

        try:
            warm_ollama_model(vlm_model, keep_alive=keep_alive)

            for pdf_path in saved_pdfs:
                status.info(f"Processing {pdf_path.name} ...")

                for page_record in iter_pdf_pages(
                    pdf_path=pdf_path,
                    pages_dir=pages_dir,
                    dpi=dpi,
                    max_pages=max_pages,
                    text_limit=text_limit,
                ):
                    try:
                        fact = analyze_page_with_fallback(
                            page_record=page_record,
                            model=vlm_model,
                            preferred_use_vision=use_vision,
                            delete_image_after=delete_images_after,
                            first_timeout=first_timeout,
                            second_timeout=second_timeout,
                            keep_alive=keep_alive,
                        )
                        append_jsonl(facts_jsonl, fact)
                        success_count += 1

                    except Exception as e:
                        failure_count += 1
                        error_record = {
                            "page_number": page_record["page"],
                            "source_pdf": page_record["source_pdf"],
                            "image_path": page_record["image_path"],
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                        append_jsonl(error_jsonl, error_record)

                    processed_count += 1
                    progress.progress(min(processed_count / max(total_pages_estimate, 1), 1.0))
                    status.info(
                        f"Processed {processed_count}/{total_pages_estimate} pages | "
                        f"success: {success_count} | failed: {failure_count}"
                    )

                    del page_record
                    gc.collect()

            status.success(
                f"Done. Facts: {success_count} pages | Failures: {failure_count} pages"
            )
            st.success("PDF processing finished.")

            preview_facts = load_jsonl(facts_jsonl)[:5]
            preview_errors = load_jsonl(error_jsonl)[:5]

            if preview_facts:
                st.subheader("Preview of page facts")
                st.json(preview_facts)

            if preview_errors:
                st.subheader("Preview of page errors")
                st.json(preview_errors)

        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.code(traceback.format_exc())


if brief_btn:
    if not st.session_state.facts_jsonl:
        st.error("Process PDFs first.")
    else:
        facts_jsonl = Path(st.session_state.facts_jsonl)
        page_facts = load_jsonl(facts_jsonl)

        if not page_facts:
            st.error("No page facts found.")
        else:
            run_dir = Path(st.session_state.run_dir)
            doc_brief_path = run_dir / "output" / "document_brief.json"

            try:
                warm_ollama_model(writer_model, keep_alive=keep_alive)

                with st.spinner("Building document brief..."):
                    doc_brief = build_doc_brief(
                        page_facts,
                        writer_model,
                        timeout=second_timeout,
                        keep_alive=keep_alive,
                    )

                with open(doc_brief_path, "w", encoding="utf-8") as f:
                    json.dump(doc_brief, f, ensure_ascii=False, indent=2)

                st.session_state.doc_brief_path = str(doc_brief_path)

                st.success(f"Document brief saved to {doc_brief_path}")
                st.subheader("Document brief")
                st.json(doc_brief)

                del page_facts
                del doc_brief
                gc.collect()

            except Exception as e:
                st.error(f"Brief generation failed: {e}")
                st.code(traceback.format_exc())


if script_btn:
    if not st.session_state.doc_brief_path:
        st.error("Build document brief first.")
    else:
        doc_brief_path = Path(st.session_state.doc_brief_path)
        run_dir = Path(st.session_state.run_dir)
        script_json_path = run_dir / "output" / "script.json"

        try:
            with open(doc_brief_path, "r", encoding="utf-8") as f:
                doc_brief = json.load(f)

            warm_ollama_model(writer_model, keep_alive=keep_alive)

            with st.spinner("Writing script..."):
                script = build_script(
                    doc_brief,
                    writer_model,
                    minutes=minutes,
                    timeout=second_timeout,
                    keep_alive=keep_alive,
                )

            with open(script_json_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=2)

            st.session_state.script_json_path = str(script_json_path)

            st.success(f"Script saved to {script_json_path}")
            st.subheader("Script preview")
            st.json(script)

            del doc_brief
            del script
            gc.collect()

        except Exception as e:
            st.error(f"Script generation failed: {e}")
            st.code(traceback.format_exc())


st.divider()
st.subheader("Generate Podcast Audio")

audio_col1, audio_col2, audio_col3 = st.columns(3)

host_a_label = None
host_b_label = None
voices = {}

try:
    voices = load_voices_json(Path(voices_json_path))
    labels = list(voices.keys())

    with audio_col1:
        host_a_label = st.selectbox("Host A voice", labels, index=0 if labels else None)

    with audio_col2:
        default_b = 1 if len(labels) > 1 else 0
        host_b_label = st.selectbox("Host B voice", labels, index=default_b if labels else None)

    with audio_col3:
        generate_audio_btn = st.button("LAST: Generate PodCast Audio")

except Exception as e:
    st.warning(f"Could not load voices.json: {e}")
    generate_audio_btn = st.button("4) Generate audio", disabled=True)


if "generate_audio_btn" in locals() and generate_audio_btn:
    if not st.session_state.script_json_path:
        st.error("Write script first.")
    else:
        run_dir = Path(st.session_state.run_dir)
        script_json_path = Path(st.session_state.script_json_path)
        output_wav = run_dir / "output" / "podcast.wav"
        output_mp3 = run_dir / "output" / "podcast.mp3" if export_mp3 else None

        try:
            with open(script_json_path, "r", encoding="utf-8") as f:
                script = json.load(f)

            turns = script.get("turns", [])
            if script.get("intro"):
                turns = [{"speaker": "Host A", "text": script["intro"]}] + turns
            if script.get("closing"):
                turns = turns + [{"speaker": "Host B", "text": script["closing"]}]

            with st.spinner("Generating audio..."):
                from tts import synthesize_dialogue_from_labels

                result = synthesize_dialogue_from_labels(
                    turns=turns,
                    host_a_label=host_a_label,
                    host_b_label=host_b_label,
                    voices_json_path=voices_json_path,
                    output_wav=output_wav,
                    lang_code=lang_code,
                    speed=speed,
                    output_mp3=output_mp3,
                )

            wav_path = Path(result["wav_path"])
            st.success(f"Audio created: {wav_path}")
            st.audio(str(wav_path), format="audio/wav")

            if result.get("mp3_path"):
                mp3_path = Path(result["mp3_path"])
                st.write(f"MP3: `{mp3_path}`")

            del script
            del turns
            gc.collect()

        except Exception as e:
            st.error(f"Audio generation failed: {e}")
            st.code(traceback.format_exc())


st.divider()
st.subheader("Run outputs")

if st.session_state.run_dir:
    run_dir = Path(st.session_state.run_dir)
    st.write(f"Current run folder: `{run_dir}`")

    if run_dir.exists():
        output_files = sorted(run_dir.rglob("*"))
        file_list = [str(p.relative_to(run_dir)) for p in output_files if p.is_file()]
        st.code("\n".join(file_list) if file_list else "No files yet.")