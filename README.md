# Assignment 04 — SFT → DPO Fine-Tuning of Qwen3-0.6B

**Track 1 (LLM) · Option A (SFT → DPO)** · BLEU + BERTScore evaluation

Group: Anushe Ali (26418), Abdullah Iqbal (26904)

## Pipeline
```
Baseline  →  SFT (5 LoRA trials → pick best)  →  DPO (5 trials → pick best)  →  Comparison
   all evaluated on the same 10 manual prompts with BLEU + BERTScore
```

- **Base model:** `Qwen/Qwen3-0.6B-Base`
- **SFT dataset:** `databricks/databricks-dolly-15k`
- **DPO dataset:** `trl-lib/ultrafeedback_binarized` (chosen/rejected pairs)
- **Fine-tuning:** LoRA adapters (fp16) for both SFT and DPO

## Results
| Stage | BLEU | BERTScore F1 |
|---|---|---|
| Base | 9.82 | 0.8803 |
| SFT (best) | 9.47 | 0.8836 |
| DPO (best) | **11.94** | **0.8863** |

- **Best SFT:** trial 3 — LoRA r=32, all attention + MLP projections, lr=1e-4, 2 epochs.
- **Best DPO:** trial 3 — beta=0.05, lr=5e-5, 1 epoch.

See `results/summary.csv` for the headline numbers and `results/*_trials.json` for per-trial scores.

## Files
| Path | What |
|---|---|
| `data/test_set.json` | 10 manual prompts with gold reference answers |
| `notebooks/00_baseline.ipynb` | Base model responses + baseline BLEU/BERTScore |
| `notebooks/01_sft.ipynb` | 5 SFT+LoRA trials, eval, auto-select best |
| `notebooks/02_dpo.ipynb` | 5 DPO+LoRA trials from best SFT, eval, select best (Colab) |
| `notebooks/02_dpo_kaggle.ipynb` | Same DPO stage adapted for the Kaggle runtime |
| `notebooks/03_comparison.ipynb` | Comparison table, bar-chart figures, qualitative examples |
| `src/eval_utils.py` | Shared eval/generation helpers (also inlined in each notebook) |
| `adapters/sft_trial{1..5}/`, `adapters/dpo_trial{1..5}/` | Saved LoRA adapters per trial |
| `results/` | Per-trial JSON, baseline JSON, and `summary.csv` |
| `figures/BLEU_by_stage.png`, `figures/BERTScore_F1_by_stage.png` | Comparison charts |
| `report/AnusheAli26418_AbdullahIqbal26904.pdf` | Final submission report — in-depth details, comparisons, and execution steps |

## How to reproduce
1. The reference answers in `data/test_set.json` are already filled (gold answers from ChatGPT/Claude/Gemini).
2. Upload `test_set.json` to your runtime (e.g. a Drive folder `MyDrive/assignment-4/` for Colab).
3. Open each notebook with a **GPU runtime** (Colab T4, or Kaggle for the `_kaggle` variant).
4. Run in order: `00 → 01 → 02 → 03`. Each saves results/adapters so a disconnect won't lose progress.
5. `03_comparison.ipynb` regenerates the tables and the `figures/*.png` used in the report.

## Notes
- Qwen3-0.6B is small enough for plain fp16 LoRA on a free T4 — no 4-bit/QLoRA needed.
- The `PROJ` path in each notebook (`/content/drive/MyDrive/assignment-4`) can be changed at the top.
- If the TRL API differs from your installed version (e.g. `processing_class` vs `tokenizer`,
  `eval_strategy` vs `evaluation_strategy`), adjust the trainer kwargs — the rest is stable.
- Deliverables: these notebooks (zip or GitHub) + the report `report/AnusheAli26418_AbdullahIqbal26904.pdf`.
