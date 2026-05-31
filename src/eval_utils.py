"""
Shared evaluation + generation utilities for Assignment 04 (Track 1, Option A).

These functions are imported by the notebooks. The notebooks also paste the key
functions inline so they remain self-contained when uploaded to Colab without the
repo. Keep the two copies in sync if you edit them.

Group: Abdullah Iqbal (26904), Anushe Ali (26418)
"""

import json
import torch

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# We use ONE consistent instruction template across the base model, the SFT
# model, and the DPO model so that all three stages are evaluated on an equal
# footing. Qwen3-0.6B-Base has no chat template, so we define our own.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Wrap a raw instruction in the training/eval prompt template."""
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def pick_dtype():
    """bf16 on Ampere+ (e.g. A100), fp16 on Turing (T4). Falls back to fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    """Greedy-decode a single response for one instruction."""
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Strip the prompt; keep only the generated continuation.
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text.strip()


def generate_all(model, tokenizer, test_set, max_new_tokens: int = 256):
    """Return list of {id, instruction, reference, response} for the test set."""
    rows = []
    for ex in test_set:
        resp = generate_response(model, tokenizer, ex["instruction"], max_new_tokens)
        rows.append({
            "id": ex["id"],
            "instruction": ex["instruction"],
            "reference": ex["reference"],
            "response": resp,
        })
    return rows


# ---------------------------------------------------------------------------
# Metrics: BLEU (sacrebleu, 0-100) and BERTScore F1 (0-1)
# ---------------------------------------------------------------------------
def compute_bleu(hypotheses, references):
    """Mean sentence-level sacreBLEU over a list of (hyp, ref) pairs. Range 0-100."""
    import sacrebleu
    scores = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hypotheses, references)]
    return sum(scores) / max(len(scores), 1)


def compute_bertscore(hypotheses, references, model_type="roberta-large"):
    """Mean BERTScore F1. Range ~0-1."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(hypotheses, references, lang="en",
                          model_type=model_type, verbose=False)
    return float(F1.mean())


def evaluate_rows(rows, bertscore_model="roberta-large"):
    """Given generated rows, return {'bleu':..., 'bertscore_f1':..., 'composite':...}."""
    hyps = [r["response"] for r in rows]
    refs = [r["reference"] for r in rows]
    bleu = compute_bleu(hyps, refs)
    bert = compute_bertscore(hyps, refs, bertscore_model)
    composite = 0.5 * (bleu / 100.0) + 0.5 * bert  # used for model selection
    return {"bleu": bleu, "bertscore_f1": bert, "composite": composite}


# ---------------------------------------------------------------------------
# Best-model selection (per assignment spec)
# ---------------------------------------------------------------------------
def select_best(trial_results, tol=0.005):
    """
    trial_results: list of dicts each with keys
        'trial', 'bleu', 'bertscore_f1', 'composite', 'val_loss'
    Rule: highest composite (0.5*BLEU/100 + 0.5*BERTScore). If two trials are
    within `tol` of the top composite, break the tie with LOWER validation loss.
    Returns the winning trial dict.
    """
    ranked = sorted(trial_results, key=lambda t: t["composite"], reverse=True)
    top = ranked[0]["composite"]
    contenders = [t for t in ranked if top - t["composite"] <= tol]
    if len(contenders) > 1:
        contenders = sorted(contenders, key=lambda t: t.get("val_loss", float("inf")))
    return contenders[0]


def load_test_set(path="data/test_set.json"):
    with open(path) as f:
        return json.load(f)
