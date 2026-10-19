# Results

Test set: 163 documents (50 bank statements, 50 invoices, 49 receipts, 14 FUNSD forms).

## Clean OCR text
| Model | Valid JSON | Exact match | Field accuracy |
|---|---|---|---|
| Stage 1 (checkpoint-376) | 89.6% | 76.7% | 91.7% |
| Final (continued adapter) | 100% | 81.0% | 94.9% |

Final adapter by document type (field accuracy / exact match):
- Invoice: 100% / 100%
- Bank statement: 98.8% / 94.0%
- Receipt: 89.8% / 71.4%
- Form: 48.9% / 0%

## Lightly perturbed OCR text (163 docs)
Valid JSON 95.1%, exact match 69.9%, field accuracy 94.4%.
Form field accuracy 28.8%, receipt 82.7%, invoice 99.3%, bank statement 97.8%.

## Limitations (please read)
1. **Forms are weak** (about 49% field accuracy on clean text). Only 14 test forms.
2. **Invoices and bank statements are synthetic templates**, so the near-100% scores show template learning, not real-world generalisation.
3. **The OCR-error test is nearly identical to the clean test** (99.5-99.95% text similarity), so it does not measure real OCR noise. Evaluation on real PaddleOCR output is still needed.
4. **Form post-processing rules were developed by inspecting the test forms**; treat those results as development-set results.
5. Possible train/test overlap for FUNSD forms was audited (`results/funsd_overlap_audit.json`); only exact-text matching was run on the test set.
6. Inference takes roughly 20-24 s per document on a T4 GPU.
