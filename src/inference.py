"""Inference helper for the fine-tuned Qwen3-14B document-extraction LoRA adapter.

The training prompt format must be kept identical at inference time:
    <instruction>\n\nOCR TEXT:\n<ocr>\n\nASSISTANT:\n<json>
"""
import json
import re

INSTRUCTION = (
    "Extract the required information from the OCR text. "
    "Correct OCR errors when necessary and return the result as JSON."
)
MAX_NEW_TOKENS = {"bank_statement": 128, "receipt": 128, "form": 512, "invoice": 640}


def build_prompt(ocr_text, instruction=INSTRUCTION):
    return f"{instruction}\n\nOCR TEXT:\n{ocr_text}\n\nASSISTANT:\n"


def load_model(adapter_path, max_seq_length=2048):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path, max_seq_length=max_seq_length,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def parse_json(text):
    text = re.sub(r"^\s*ASSISTANT:\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract(model, tokenizer, ocr_text, instruction=INSTRUCTION, max_new_tokens=640):
    import torch
    prompt = build_prompt(ocr_text, instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text, parse_json(text)
