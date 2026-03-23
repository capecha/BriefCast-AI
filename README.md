# BriefCast-AI: Local PDF-to-Podcast with Ollama
<img width="1264" height="1320" alt="image" src="https://github.com/user-attachments/assets/5387ac00-84ca-4be5-baba-8802abe238ac" />

Turn PDF documents into a friendly two-speaker podcast **fully on your machine**.

This project ingests one or more PDFs, renders each page as an image, uses a **multimodal Ollama model** to understand text + charts + infographics, builds a grounded fact pack, writes a conversation between two hosts in plain language, and finally synthesizes the dialogue into audio with **locally generated voices**.

It is designed for document-heavy workflows such as:

- economic reports
- policy briefs
- technical notes
- slide decks exported as PDF
- reports with charts, tables, diagrams, and infographics

## What the app does

1. Upload one or more PDFs.
2. Render each page to a high-resolution image.
3. Extract page text.
4. Send **page image + page text** to a multimodal Ollama model.
5. Build a structured page-level fact pack.
6. Merge all page facts into a document brief.
7. Generate a **two-person podcast script** in easy, friendly language.
8. Convert each speaker turn to speech with local TTS.
9. Merge all turns into a final WAV file.
10. Preview the result in the app.

## Why this architecture

A PDF with infographics is not just text. A text-only parser will miss chart meaning, labels, and layout cues. This project therefore uses a **multimodal pipeline**:

- **PyMuPDF** renders each PDF page to PNG and extracts text.
- **Ollama vision model** reads each page image plus extracted text.
- **Ollama text model** writes the final dialogue.
- **Kokoro** generates the audio locally.
- **Streamlit** provides the local UI.

This design is more reliable than asking one model to do everything in a single step.

---

## Recommended stack

### Core libraries

- Python 3.10+
- Streamlit
- PyMuPDF
- Pillow
- requests
- pydantic
- numpy
- soundfile
- Kokoro

### Local model runtime

- Ollama

### Recommended Ollama models

- **Page understanding:** `qwen2.5vl:7b`
- **Script generation:** `qwen3.5:9b`
- **Optional embeddings for retrieval:** `embeddinggemma`

You can swap the models later if you want faster or higher-quality variants.

---

## Project structure

```text
podcast_local/
├── app.py
├── voices.json
├── requirements.txt
├── data/
│   ├── uploads/
│   ├── rendered_pages/
│   ├── facts/
│   └── output/
└── README.md
```

### Folder purpose

- `uploads/`: original PDF files
- `rendered_pages/`: PNG version of each page
- `facts/`: extracted structured summaries per page or document
- `output/`: generated WAV/MP3 files
- `voices.json`: voice selector mapping shown in the UI

---

## End-to-end workflow

### Stage 1. PDF ingestion

For each uploaded PDF, the app:

- opens the PDF with PyMuPDF
- extracts page text
- renders each page at a higher DPI for better chart and infographic readability

### Stage 2. Page-level multimodal understanding

Each page is processed with a multimodal Ollama model.

Input sent to the model:

- page image
- extracted text
- prompt instructing the model to identify:
  - title
  - main findings
  - important numbers
  - chart takeaways
  - jargon to explain
  - caveats / uncertain readings

Output expected from the model:

- structured JSON

### Stage 3. Document brief

The page-level fact packs are merged into a single grounded brief that includes:

- executive summary
- major themes
- key statistics
- policy relevance
- terms that need explanation

### Stage 4. Podcast script generation

A second Ollama model turns the grounded document brief into:

- episode title
- intro hook
- turn-by-turn script for Host A and Host B
- closing summary

The writing style should be:

- friendly
- clear
- plain language
- accurate
- non-robotic

### Stage 5. Local TTS

Each line is synthesized locally:

- Host A uses Voice A
- Host B uses Voice B

Then all audio chunks are concatenated into one final WAV file.

---

## Installation

### 1. Install system dependencies

#### macOS

```bash
brew install ollama ffmpeg espeak-ng
```

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y ffmpeg espeak-ng
```

Install Ollama from its official installer if it is not already available.

---

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Python packages

```bash
pip install --upgrade pip
pip install streamlit pymupdf pillow requests pydantic numpy soundfile kokoro
```

If you plan to use additional language support in Kokoro:

```bash
pip install "misaki[ja]"   # Japanese
pip install "misaki[zh]"   # Mandarin Chinese
```

---

### 4. Pull Ollama models

```bash
ollama pull qwen2.5vl:7b
ollama pull qwen3.5:9b
ollama pull embeddinggemma
```

---

## Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

---

## Example configuration for `voices.json`

```json
{
  "Warm Female": "af_heart",
  "Clear Female": "af_bella",
  "Warm Male": "am_adam",
  "Calm Male": "bm_george"
}
```

The exact voice IDs available in your environment may vary depending on the Kokoro package version and your voice assets.

---

## Minimal example: render a PDF page

```python
import fitz
from pathlib import Path

def render_pdf(pdf_path: str, out_dir: str, dpi: int = 240):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_path = out_dir / f"page_{i+1}.png"
        img_path.write_bytes(pix.tobytes("png"))
        pages.append({
            "page": i + 1,
            "text": text,
            "image_path": str(img_path),
        })

    return pages
```

---

## Minimal example: call Ollama for structured JSON

```python
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "main_points": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["title", "main_points"]
}

payload = {
    "model": "qwen3.5:9b",
    "messages": [
        {"role": "user", "content": "Summarize this page as JSON."}
    ],
    "stream": False,
    "format": schema
}

response = requests.post(OLLAMA_URL, json=payload, timeout=120)
response.raise_for_status()
data = response.json()
parsed = json.loads(data["message"]["content"])
print(parsed)
```

---

## Minimal example: Kokoro TTS

```python
from kokoro import KPipeline
import numpy as np
import soundfile as sf

pipeline = KPipeline(lang_code='a')  # American English
text = "Welcome to today's episode. We are discussing what this report means in simple language."

segments = []
for _, _, audio in pipeline(text, voice='af_heart', speed=1.0):
    segments.append(audio.astype('float32'))

final_audio = np.concatenate(segments)
sf.write("sample.wav", final_audio, 24000)
```

---

## Recommended prompt rules for page extraction

Use prompts that force the model to stay grounded. Good rules include:

- use both the image and the extracted text
- do not invent facts not visible in the page
- identify exact numbers when possible
- explain chart meaning in simple language
- mention uncertainty if a graphic is hard to read
- return only JSON matching the schema

---

## Recommended prompt rules for script generation

For the second-stage writer model, specify:

- two hosts
- friendly tone
- easy language for non-specialists
- short speaking turns
- explain jargon immediately
- only use facts from the document brief
- mention uncertainty when the source is ambiguous
- close with a short summary of takeaways

---

## Performance notes

- Rendering pages at **220–300 DPI** improves chart reading but increases processing time.
- Use a **smaller vision model** if you want faster page extraction.
- Use a **larger writer model** if you want better conversational quality.
- If a document is very large, cache the page-level JSON so you do not re-run the entire extraction every time.
- For multiple documents, add embeddings and retrieval so the script is grounded only on the most relevant passages.

---

## Troubleshooting

### Ollama is not responding

Check that the server is running:

```bash
ollama serve
```

Then test locally:

```bash
curl http://localhost:11434/api/tags
```

### The model misses chart details

Try one or more of these:

- increase render DPI to 300
- crop important chart regions and re-run extraction
- reduce the amount of text sent with the page
- add a second pass focused only on visuals

### Kokoro fails on macOS Apple Silicon

Try running with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py
```

### TTS sounds unnatural

Try:

- shorter turns
- more punctuation in the script
- a different voice pair
- slower speaking speed
- cleaning up long numbers, acronyms, and formulas before synthesis

### Spanish pronunciation is weak

Use the correct Kokoro language pipeline for Spanish and make sure the voice and language code match.

---

## Suggested roadmap

### v1

- upload PDFs
- multimodal extraction
- structured fact pack
- two-host script
- two selectable voices
- final WAV output

### v2

- edit script before synthesis
- add source citations by page number
- save projects
- export MP3
- chapter markers
- retry per page

### v3

- multi-document retrieval with embeddings
- bilingual podcasts
- optional narrator mode
- per-speaker speed and pause control
- glossary mode for technical terms

---

## Privacy

This project is designed to run locally:

- local PDFs
- local Ollama inference
- local TTS
- local Streamlit interface

That makes it suitable for sensitive internal reports, drafts, and working papers.

---

## License

Choose the license that fits your project, for example MIT.

---

## A good first milestone

Before building the full UI, validate the following in order:

1. Render a PDF page correctly.
2. Extract a good JSON summary from one page.
3. Generate a short script from two or three page summaries.
4. Generate clean speech from two different Kokoro voices.
5. Only then connect the full Streamlit workflow.

That sequence will save a lot of debugging time.

---

# Kokoro TTS setup and `tts.py` integration

This section replaces the earlier TTS snippet and aligns the project with Kokoro's current behavior, where generated audio can arrive as a PyTorch tensor. The code now converts audio safely before concatenating and writing files.

## Install TTS dependencies

On macOS:

```bash
brew install ffmpeg espeak-ng
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install kokoro soundfile numpy
```

## Create `voices.json`

Put this file in the project root:

```json
{
  "Warm Female": "af_heart",
  "Clear Female": "af_bella",
  "Warm Male": "am_adam",
  "Calm Male": "bm_george"
}
```

You can change the voice IDs later after testing which ones you prefer.

## Add `tts.py`

Place the provided `tts.py` file in the project root. It does four things:

1. Loads the user-facing voice labels from `voices.json`
2. Maps Host A and Host B to Kokoro voice IDs
3. Synthesizes each turn of the script with Kokoro
4. Saves a WAV file and optionally converts it to MP3 with `ffmpeg`

## Minimal usage example

```python
from pathlib import Path
from tts import synthesize_dialogue_from_labels

turns = [
    {"speaker": "Host A", "text": "Welcome to our podcast."},
    {"speaker": "Host B", "text": "Today we explain this report in plain language."},
]

result = synthesize_dialogue_from_labels(
    turns=turns,
    host_a_label="Warm Female",
    host_b_label="Calm Male",
    voices_json_path="voices.json",
    output_wav=Path("data/output/podcast.wav"),
    lang_code="a",
    speed=1.0,
    output_mp3=Path("data/output/podcast.mp3"),
)

print(result)
```

Expected output:

```python
{
    "wav": "data/output/podcast.wav",
    "mp3": "data/output/podcast.mp3"
}
```

## Streamlit integration example

```python
import json
import streamlit as st
from pathlib import Path
from tts import synthesize_dialogue_from_labels

with open("voices.json", "r", encoding="utf-8") as f:
    voices = json.load(f)

voice_a_label = st.selectbox("Voice for Host A", list(voices.keys()), index=0)
voice_b_label = st.selectbox("Voice for Host B", list(voices.keys()), index=1)

script = {
    "turns": [
        {"speaker": "Host A", "text": "Welcome to the episode."},
        {"speaker": "Host B", "text": "We will break this document down simply."},
    ]
}

if st.button("Generate audio"):
    result = synthesize_dialogue_from_labels(
        turns=script["turns"],
        host_a_label=voice_a_label,
        host_b_label=voice_b_label,
        voices_json_path="voices.json",
        output_wav=Path("data/output/podcast.wav"),
        lang_code="a",
        speed=1.0,
        output_mp3=Path("data/output/podcast.mp3"),
    )

    st.audio(result["wav"], format="audio/wav")
    st.success(f"Saved: {result}")
```

## Apple Silicon note

If you hit MPS issues on an M-series Mac, try:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 streamlit run app.py
```

or for a direct test:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_kokoro.py
```

## Why the earlier code failed

The earlier TTS snippet used:

```python
audio.astype(np.float32)
```

That fails when `audio` is a PyTorch tensor. The updated `tts.py` fixes this by converting tensors safely with:

```python
if hasattr(audio, "detach"):
    audio = audio.detach().cpu().numpy()
```

before calling `np.asarray(audio, dtype=np.float32)`.
