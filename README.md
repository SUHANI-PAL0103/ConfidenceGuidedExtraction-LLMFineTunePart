# Confidence-Guided Document Information Extraction: Fine-tuned LLM

B.Tech CSE project, Graphic Era Hill University (Team CSE27-019). Authors: Suhani Pal, Anshika Panwar, Mansi Bhandari.

This repository contains the **fine-tuned LLM component**: Qwen3-14B-Base fine-tuned with QLoRA
(Unsloth, r=16) to turn OCR text into structured JSON for receipts, invoices, bank statements and forms.
The confidence-based router (selective LLM invocation) is a separate module.

## Model weights
LoRA adapter (245 MB, too large for GitHub): https://huggingface.co/Suhani0103/qwen3-14b-doc-extraction-lora

## Layout
- `src/inference.py` – canonical prompt format, model loading, JSON parsing
- `src/postprocess_forms.py` – rule-based post-processing for forms (Hybrid v3)
- `src/summarize_results.py` – summarise a results file
- `data/` – SFT train/validation, clean test, OCR-error test, form continuation set
- `results/` – raw per-document predictions and the FUNSD overlap audit
- `requirements.txt` – exact package versions used (Colab T4)

## Training
- Stage 1: 1,502 examples, 2 epochs (376 steps), lr 2e-4, final training loss 0.833
- Stage 2 (form-focused continuation): 177 examples (59 new forms + 118 retention), 2 epochs (46 steps), lr 5e-5, loss 0.799
- Hardware: Tesla T4 (16 GB), 4-bit QLoRA, max sequence length 1536

## Quick start
```python
from src import inference
model, tok = inference.load_model("path/or/hf-repo-of-adapter")
text, parsed = inference.extract(model, tok, ocr_text)
```

See `RESULTS.md` for results and limitations.
