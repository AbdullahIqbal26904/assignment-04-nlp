# Assignment 04 — SFT → DPO Fine-Tuning of Qwen3-0.6B

**Track 1 (LLM) · Option A (SFT → DPO)** · BLEU + BERTScore evaluation

Group: Abdullah Iqbal (26904), Anushe Ali (26418)

## Pipeline
```
Baseline  →  SFT (5 LoRA trials → pick best)  →  DPO (5 trials → pick best)  →  Comparison
   all evaluated on the same 10 manual prompts with BLEU + BERTScore
```

## Files
| Path | What |
|---|---|
| `data/test_set.json` | 10 manual prompts — **fill the `reference` fields with gold answers first** |
| `notebooks/00_baseline.ipynb` | Base model responses + baseline BLEU/BERTScore |
| `notebooks/01_sft.ipynb` | 5 SFT+LoRA trials, eval, auto-select best |
| `notebooks/02_dpo.ipynb` | 5 DPO+LoRA trials from best SFT, eval, select best |
| `notebooks/03_comparison.ipynb` | Comparison table, bar-chart figures, qualitative examples |
| `src/eval_utils.py` | Shared eval/generation helpers (also inlined in each notebook) |
| `report/REPORT_TEMPLATE.md` | Report scaffold — fill placeholders, export to PDF/DOCX |
| `build_notebooks.py` | Regenerates the notebooks (only needed if you edit cell content) |

## How to run (Google Colab)
1. **Fill `data/test_set.json`** — paste gold reference answers from ChatGPT/Claude/Gemini.
2. Create a Drive folder `MyDrive/assignment-4/` and upload `test_set.json` into it.
3. Open each notebook in Colab with a **GPU runtime** (Runtime → Change runtime type → T4).
4. Run in order: `00 → 01 → 02 → 03`. Each saves results/adapters to Drive, so a disconnect won't lose progress.
5. Copy the printed tables, the `figures/*.png`, and qualitative examples into `report/REPORT_TEMPLATE.md`.
6. Export the report to PDF/DOCX named **`Abdullah_Anushe.pdf`** and submit on LMS.

## Notes
- Qwen3-0.6B is small enough for plain fp16 LoRA on a free T4 — no 4-bit/QLoRA needed.
- The `PROJ` path in each notebook (`/content/drive/MyDrive/assignment-4`) can be changed at the top.
- If the TRL API differs from your installed version (e.g. `processing_class` vs `tokenizer`,
  `eval_strategy` vs `evaluation_strategy`), adjust the trainer kwargs — the rest is stable.
- Deliverables: these notebooks (zip or GitHub) + the report.
