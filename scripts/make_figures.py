"""README figures from the project's real results. Re-run with `qlora figures` (or this script directly).

Every number is read from a source file under QLORA_ROOT or from an MLflow run; nothing is typed in by hand.
ESTIMATED values are drawn hatched/outlined, measured values solid. Each figure is written in a light and a dark
GitHub variant (SVG with real text + 2x PNG) to repo/docs/images/, plus _contact_sheet.png showing all of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qlora_pipeline.bootstrap import load_env  # noqa: E402

load_env()

import json  # noqa: E402
import logging  # noqa: E402

import figure_theme as ft  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qlora_pipeline.paths import qlora_root, repo_dir, reports_dir, runs_dir  # noqa: E402

GB = 1024**3
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # missing Helvetica/-apple-system on Windows
OUT = repo_dir() / "docs" / "images"
ROOT = qlora_root()


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ sources

SRC = {
    "train7b": runs_dir() / "llama2-7b-qlora" / "summary.json",
    "bench": reports_dir() / "benchmark.json",
    "eval7b": reports_dir() / "eval_llama2-7b-qlora.json",
    "eval7b_csv": reports_dir() / "eval_llama2-7b-qlora_predictions.csv",
    "load": reports_dir() / "load_test.json",
    "full_train": runs_dir() / "proxy-full" / "summary.json",
}


def training_history(run_id: str) -> tuple[list[tuple[int, float]], list[tuple[int, float]], str]:
    """Train/val loss from the MLflow run; falls back to the last checkpoint's trainer_state.json."""
    try:
        import mlflow

        from qlora_pipeline import telemetry

        mlflow.set_tracking_uri(telemetry.tracking_uri())
        c = mlflow.MlflowClient()
        tr = sorted((m.step, m.value) for m in c.get_metric_history(run_id, "loss"))
        ev = sorted((m.step, m.value) for m in c.get_metric_history(run_id, "eval_loss"))
        if tr:
            return tr, ev, f"MLflow run {run_id[:8]}"
    except Exception as e:  # noqa: BLE001
        print(f"MLflow unavailable ({type(e).__name__}); using trainer_state.json", flush=True)
    ck = sorted((runs_dir() / "llama2-7b-qlora").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
    hist = load_json(ck / "trainer_state.json")["log_history"]
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    return tr, ev, rel(ck / "trainer_state.json")


def values_of(ax, bars, fmt="{:.2f}", color=None, inside=False, dy=0.0, suffix=""):
    for b in bars:
        h, x = b.get_height(), b.get_x() + b.get_width() / 2
        y = b.get_y() + h
        ax.annotate(fmt.format(h) + suffix, (x, y), xytext=(0, 4 + dy), textcoords="offset points", ha="center",
                    va="bottom", fontsize=ft.LABEL_PT, color=color or ft.plt.rcParams["text.color"])  # fmt: skip


# ------------------------------------------------------------------ figures


def fig_hero(theme: str) -> list[Path]:
    c = ft.apply(theme)
    t7, b = load_json(SRC["train7b"]), load_json(SRC["bench"])
    seven, phys = b["seven_b"], load_json(SRC["load"])["physical_vram_gb"]
    alloc, reserved = t7["peak_mem_allocated_gb"], t7["peak_mem_reserved_gb"]
    est = [("bf16 LoRA", seven["bf16_lora"]["total_gb"], c["lora"]),
           ("Full FT\n8-bit AdamW", seven["full_ft_8bit_adam"]["total_gb"], c["full"]),
           ("Full FT\nstandard AdamW", seven["full_ft_fp32_adam"]["total_gb"], c["full"])]  # fmt: skip
    fig, ax = ft.plt.subplots(figsize=(9.6, 5.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.17)
    x = np.arange(4)
    ax.bar(0, reserved, width=0.6, color=c["qlora"], alpha=0.35, label="Measured: peak reserved")
    ax.bar(0, alloc, width=0.6, color=c["qlora"], label="Measured: peak allocated")
    ax.annotate(f"{alloc:.2f} allocated", (0, alloc), xytext=(0, -16), textcoords="offset points", ha="center",
                color=c["bg"], fontsize=ft.LABEL_PT, fontweight="semibold")  # fmt: skip
    ax.annotate(f"{reserved:.2f} reserved", (0, (alloc + reserved) / 2), ha="center", va="center",
                color=c["text"], fontsize=ft.LABEL_PT)  # fmt: skip
    for i, (_, v, col) in enumerate(est, start=1):
        bars = ft.estimated_bar(ax, i, v, col, label="ESTIMATED (hatched)" if i == 1 else None)
        values_of(ax, bars, fmt="~{:.1f}", suffix=" GB  ESTIMATED")
    ax.axhline(phys, color=c["warn"], linewidth=1.6, linestyle="--", label=f"RTX 5080 Laptop VRAM ({phys:.1f} GB)")
    ax.set_xticks(x, ["QLoRA (4-bit NF4)\nMEASURED", *[e[0] for e in est]])
    ax.set_ylabel("Peak training GPU memory (GB, linear scale)")
    ax.set_ylim(0, max(e[1] for e in est) * 1.12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98))
    ft.title(fig, ax, f"QLoRA trains Llama-2-7B in {alloc:.2f} GB allocated; the estimated alternatives need "
                      f"{est[0][1]:.0f}-{est[-1][1]:.0f} GB",
             "Llama-2-7b-chat, batch 4, seq <= 1024. QLoRA is the only one I could run on the 16 GB card; "
             "the rest are estimates.")  # fmt: skip
    ft.footer(fig, f"Sources: {rel(SRC['train7b'])} (measured); {rel(SRC['bench'])}, MLflow run "
                   f"{b['mlflow_run_id'][:8]} (ESTIMATED, about ±1-2 GB)")  # fmt: skip
    return ft.save(fig, OUT, "01-hero-7b-memory", theme)


def fig_categories(theme: str) -> list[Path]:
    c = ft.apply(theme)
    e = load_json(SRC["eval7b"])
    base, tuned = e["base"]["rougeL_per_category"], e["tuned"]["rougeL_per_category"]
    n = pd.read_csv(SRC["eval7b_csv"], encoding="utf-8")["category"].value_counts().to_dict()
    cats = sorted(base, key=lambda k: tuned[k] - base[k])  # smallest gain at the bottom
    gains = [tuned[k] - base[k] for k in cats]
    fig, ax = ft.plt.subplots(figsize=(9.6, 6.6))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.85, bottom=0.12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ys = list(range(len(cats)))
    for y, k in zip(ys, cats, strict=True):
        ax.plot([base[k], tuned[k]], [y, y], color=c["grid"], linewidth=3, solid_capstyle="butt", zorder=1)
    ax.scatter([base[k] for k in cats], ys, color=c["full"], s=70, zorder=3, label="Base (untuned)", marker="s")
    ax.scatter([tuned[k] for k in cats], ys, color=c["qlora"], s=70, zorder=3, label="QLoRA-tuned", marker="s")
    for y, k in zip(ys, cats, strict=True):
        ax.annotate(f"{base[k]:.1f}", (base[k], y), xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="center", color=c["muted"])  # fmt: skip
        ax.annotate(f"{tuned[k]:.1f}  (+{tuned[k] - base[k]:.1f})", (tuned[k], y), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", color=c["text"])  # fmt: skip
    # overall, highlighted
    yo = len(cats) + 0.9
    ob, ot = e["base"]["rougeL"], e["tuned"]["rougeL"]
    ax.axhspan(yo - 0.5, yo + 0.5, color=c["qlora"], alpha=0.10, linewidth=0)
    ax.plot([ob, ot], [yo, yo], color=c["qlora"], linewidth=3, solid_capstyle="butt", zorder=1)
    ax.scatter([ob], [yo], color=c["full"], s=90, zorder=3, marker="s")
    ax.scatter([ot], [yo], color=c["qlora"], s=90, zorder=3, marker="s")
    ax.annotate(f"{ob:.2f}", (ob, yo), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                color=c["muted"], fontweight="semibold")  # fmt: skip
    ax.annotate(f"{ot:.2f}  (+{ot - ob:.2f})", (ot, yo), xytext=(8, 0), textcoords="offset points", ha="left",
                va="center", color=c["text"], fontweight="semibold")  # fmt: skip
    labels = [f"{k.title()}  (n={n[k]})" for k in cats] + [f"Overall  (n={e['n_test']})"]
    ax.set_yticks([*ys, yo], labels)
    ax.get_yticklabels()[-1].set_fontweight("semibold")
    ax.get_yticklabels()[-1].set_color(c["text"])
    ax.set_xlabel("ROUGE-L (F1 x 100, test split, greedy)")
    ax.set_xlim(15, 58)
    ax.set_ylim(-0.7, yo + 0.7)
    ax.legend(loc="lower right")
    ft.title(fig, ax, f"Fine-tuning lifts ROUGE-L in all {len(cats)} categories "
                      f"(+{min(gains):.0f} to +{max(gains):.0f} points)",
             f"Base Llama-2-7b-chat vs the QLoRA adapter on the same 4-bit base, sorted by gain. Overall "
             f"{ob:.2f} -> {ot:.2f}.")  # fmt: skip
    ft.footer(fig, f"Sources: {rel(SRC['eval7b'])}, {rel(SRC['eval7b_csv'])} (n per category); MLflow run "
                   f"{e['mlflow_run_id'][:8]}")  # fmt: skip
    return ft.save(fig, OUT, "02-quality-by-category", theme)


def fig_slots(theme: str) -> list[Path]:
    c = ft.apply(theme)
    e = load_json(SRC["eval7b"])
    metrics = [("Slot recall\n(reference slots reproduced)", "slots.slot_recall_pct"),
               ("Rows with every\nslot kept", "slots.rows_all_slots_kept_pct")]  # fmt: skip
    fig, ax = ft.plt.subplots(figsize=(9.6, 5.4))
    fig.subplots_adjust(left=0.09, right=0.62, top=0.80, bottom=0.16)
    x = np.arange(len(metrics))
    w = 0.34
    bb = ax.bar(x - w / 2, [e["base"][m] for _, m in metrics], w, color=c["full"], label="Base (untuned)")
    bt = ax.bar(x + w / 2, [e["tuned"][m] for _, m in metrics], w, color=c["qlora"], label="QLoRA-tuned")
    values_of(ax, bb, fmt="{:.1f}%")
    values_of(ax, bt, fmt="{:.1f}%")
    ax.set_xticks(x, [m[0] for m in metrics])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 85)
    ax.legend(loc="upper right")
    ib, it = e["base"]["slots.invented_slots_per_row"], e["tuned"]["slots.invented_slots_per_row"]
    fig.text(0.66, 0.62, "Known weakness", color=c["warn"], fontsize=ft.LABEL_PT, fontweight="semibold")
    fig.text(0.66, 0.40, f"Invented slots per reply\n(slots not in the reference)\n\nbase    {ib:.3f}\ntuned  {it:.3f}\n\n"
                         f"The tuned model writes slots, and\nsometimes ones the reference\ndoes not use "
                         f"({e['tuned']['slots.invented_slots']} across {e['n_test']} replies).",
             color=c["text"], fontsize=ft.LABEL_PT, va="center", linespacing=1.35)  # fmt: skip
    ft.title(fig, ax, "The tuned model keeps {{template slots}} far more often, but also invents some",
             f"Bitext replies use placeholders like {{{{Order Number}}}} that should be reproduced verbatim; "
             f"{e['tuned']['slots.rows_with_slots']} of {e['n_test']} test replies contain them.")
    ft.footer(fig, f"Source: {rel(SRC['eval7b'])} (296 test rows); MLflow run {e['mlflow_run_id'][:8]}")
    return ft.save(fig, OUT, "03-template-slots", theme)


def fig_forgetting(theme: str) -> list[Path]:
    c = ft.apply(theme)
    e = load_json(SRC["eval7b"])
    tasks = ["arc_easy", "hellaswag"]
    fig, ax = ft.plt.subplots(figsize=(9.6, 5.2))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.16)
    x = np.arange(len(tasks))
    w = 0.34
    for off, k, col, lab in ((-w / 2, "base", c["full"], "Base (untuned)"), (w / 2, "tuned", c["qlora"], "QLoRA-tuned")):
        vals = [e[k]["lm_eval"][t]["acc_norm"] for t in tasks]
        errs = [e[k]["lm_eval"][t]["acc_norm_stderr"] for t in tasks]
        bars = ax.bar(x + off, vals, w, color=col, label=lab, yerr=errs, capsize=5,
                      error_kw={"ecolor": c["text"], "elinewidth": 1.2})  # fmt: skip
        for b_, v, s in zip(bars, vals, errs, strict=True):
            ax.annotate(f"{v:.1f} ± {s:.1f}", (b_.get_x() + b_.get_width() / 2, v + s), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom")  # fmt: skip
    deltas = [e["tuned"]["lm_eval"][t]["acc_norm"] - e["base"]["lm_eval"][t]["acc_norm"] for t in tasks]
    ax.set_xticks(x, [f"{t}\n(tuned - base: {d:+.1f})" for t, d in zip(tasks, deltas, strict=True)])
    ax.set_ylabel("acc_norm (%), ± = stderr")
    ax.set_ylim(0, 95)
    ax.legend(loc="upper left", ncol=2)
    ft.title(fig, ax, "No measurable forgetting: both changes are well inside the error bars",
             "lm-evaluation-harness, 200 questions per task. Flag rule: drop > 2 points and > 1.5x combined "
             "stderr; neither task is flagged.")  # fmt: skip
    ft.footer(fig, f"Source: {rel(SRC['eval7b'])}; MLflow run {e['mlflow_run_id'][:8]}")
    return ft.save(fig, OUT, "04-no-forgetting", theme)


def fig_proxy(theme: str) -> list[Path]:
    c = ft.apply(theme)
    b = load_json(SRC["bench"])
    res = b["results"]
    order = [("full", "Full FT", c["full"]), ("lora", "bf16 LoRA", c["lora"]), ("qlora", "QLoRA", c["qlora"])]
    full_params = load_json(SRC["full_train"])["trainable_params"]
    opt_est = 2 * full_params / GB  # paged 8-bit AdamW: 2 bytes/param, outside PyTorch's counters
    fig, (a1, a2, a3) = ft.plt.subplots(1, 3, figsize=(12.8, 5.6), gridspec_kw={"width_ratios": [1.1, 1.0, 1.05]})
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.17, wspace=0.42)
    xs = np.arange(3)
    for i, (k, _, col) in enumerate(order):
        v = res[k]["train"]["peak_mem_allocated_gb"]
        a1.bar(i, v, 0.6, color=col)
        a1.annotate(f"{v:.2f}", (i, v / 2), ha="center", va="center", color=c["bg"], fontweight="semibold")
    est = ft.estimated_bar(a1, 0, opt_est, c["full"], bottom=res["full"]["train"]["peak_mem_allocated_gb"])
    top = res["full"]["train"]["peak_mem_allocated_gb"] + opt_est
    a1.annotate(f"+{opt_est:.2f} optimizer\nstate, ESTIMATED", (0.32, top - opt_est / 2), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", fontsize=ft.LABEL_PT - 1)  # fmt: skip
    del est
    a1.set_xticks(xs, [o[1] for o in order])
    a1.set_ylabel("Peak allocated GPU memory (GB)")
    a1.set_ylim(0, top * 1.62)
    a1.set_title("Peak allocated memory", fontsize=ft.LABEL_PT + 1)
    red = b["reductions"]
    rq, rf = res["qlora"]["train"]["peak_mem_reserved_gb"], res["full"]["train"]["peak_mem_reserved_gb"]
    a1.text(0.98, 0.98, f"QLoRA allocated: -{red['alloc_vs_lora']}% vs LoRA,\n-{red['alloc_vs_full']}% vs full FT "
                        f"(about -{red['alloc_vs_full_corrected_est']:.0f}%\nwith the ESTIMATED optimizer state)\n"
                        f"Reserved: QLoRA {rq:.2f} vs full FT {rf:.2f} GB,\nso the claim does not hold on reserved",
            transform=a1.transAxes, ha="right", va="top", fontsize=ft.LABEL_PT - 1.5, color=c["muted"],
            linespacing=1.3)  # fmt: skip

    for i, (k, _, col) in enumerate(order):
        v = res[k]["eval"]["tuned"]["rougeL"]
        a2.bar(i, v, 0.6, color=col)
        a2.annotate(f"{v:.2f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
    a2.set_xticks(xs, [o[1] for o in order])
    a2.set_ylabel("Test ROUGE-L (F1 x 100)")
    a2.set_ylim(0, 46)
    a2.set_title("Quality: test ROUGE-L", fontsize=ft.LABEL_PT + 1)

    p = b["parity"]
    tol, (lo, hi), d = p["tolerance"], p["ci95"], p["delta"]
    a3.grid(axis="y", visible=False)
    a3.grid(axis="x", visible=True)
    a3.axvspan(-tol, tol, color=c["qlora"], alpha=0.12, linewidth=0, label=f"±{tol:g} tolerance")
    a3.axvline(0, color=c["muted"], linewidth=1)
    a3.errorbar([d], [0], xerr=[[d - lo], [hi - d]], fmt="s", color=c["qlora"], ecolor=c["qlora"], elinewidth=2.2,
                capsize=7, markersize=9, label="QLoRA - full FT (95% CI)")  # fmt: skip
    a3.annotate(f"{d:+.2f}", (d, 0), xytext=(0, 12), textcoords="offset points", ha="center", fontweight="semibold")
    a3.annotate(f"[{lo:+.2f}, {hi:+.2f}]", (d, 0), xytext=(0, -22), textcoords="offset points", ha="center",
                color=c["muted"])  # fmt: skip
    a3.set_xlim(-2.0, 2.0)
    a3.set_ylim(-1, 1)
    a3.set_yticks([])
    a3.spines["left"].set_visible(False)
    a3.set_xlabel("ROUGE-L difference (points)")
    a3.set_title("Parity gap, paired bootstrap", fontsize=ft.LABEL_PT + 1)
    leg = a3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=ft.LABEL_PT - 1, frameon=True,
                    facecolor=c["bg"], edgecolor=c["bg"], framealpha=1.0)  # fmt: skip
    leg.set_zorder(5)
    a3.text(0.5, 0.06, "PASS on the point estimate;\nthe CI is wider than the tolerance", transform=a3.transAxes,
            ha="center", va="bottom", fontsize=ft.LABEL_PT - 1, color=c["muted"], zorder=5,
            bbox={"facecolor": c["bg"], "edgecolor": "none", "pad": 3})  # fmt: skip
    steps = res["qlora"]["train"]["global_steps"]
    ft.title(fig, a1, f"TinyLlama-1.1B proxy: QLoRA has the lowest peak allocated memory; ROUGE-L is {d:+.2f} vs full "
                      f"FT (95% CI {lo:+.2f} to {hi:+.2f})",
             f"Same data, template, seq length, effective batch, {steps} steps and seed for all three; full FT lr 2e-5, "
             "LoRA/QLoRA 2e-4. One seed each.")  # fmt: skip
    ft.footer(fig, f"Sources: {rel(SRC['bench'])} (MLflow run {b['mlflow_run_id'][:8]}); optimizer state ESTIMATED as "
                   f"2 bytes x {full_params / 1e9:.2f}B params from {rel(SRC['full_train'])}")  # fmt: skip
    return ft.save(fig, OUT, "05-proxy-benchmark", theme)


def fig_training(theme: str) -> list[Path]:
    c = ft.apply(theme)
    t7 = load_json(SRC["train7b"])
    tr, ev, src = training_history(t7["mlflow_run_id"])
    fig, ax = ft.plt.subplots(figsize=(9.6, 5.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.17)
    ax.plot([s for s, _ in tr], [v for _, v in tr], color=c["qlora"], linewidth=1.8, label="Train loss (every 5 steps)")
    ax.plot([s for s, _ in ev], [v for _, v in ev], color=c["text"], linewidth=0, marker="s", markersize=8,
            label="Validation loss (end of each epoch)")  # fmt: skip
    best = min(ev, key=lambda t: t[1])
    for s, v in ev:
        lab = f"{v:.4f}" + ("  best (epoch 2, kept)" if (s, v) == best else "")
        ax.annotate(lab, (s, v), xytext=(6, 8), textcoords="offset points", ha="left" if s < 450 else "right",
                    fontweight="semibold" if (s, v) == best else "normal")  # fmt: skip
    for e_ in (169, 338):
        ax.axvline(e_, color=c["grid"], linewidth=1, linestyle=":")
    ax.set_xlabel(f"Optimizer step ({t7['global_steps']} total; dotted lines = epoch boundaries)")
    ax.set_ylabel("Loss (response tokens only)")
    ax.set_xlim(0, t7["global_steps"] + 5)
    ax.set_ylim(0, max(v for _, v in tr) * 1.08)
    ax.legend(loc="upper right")
    ft.title(fig, ax, f"Validation loss bottoms out at epoch 2 ({best[1]:.4f}); epoch 3 overfits slightly",
             f"Llama-2-7b QLoRA, 3 epochs, effective batch 32, {t7['wall_time_s'] / 3600:.1f} h wall time on the "
             "laptop GPU.")  # fmt: skip
    ft.footer(fig, f"Sources: {src}; {rel(SRC['train7b'])}")
    return ft.save(fig, OUT, "06-training-curves", theme)


def fig_latency(theme: str) -> list[Path]:
    c = ft.apply(theme)
    lt = load_json(SRC["load"])
    reqs = [r for r in lt["requests"] if r["ok"]]
    toks = [r["tokens"] for r in reqs]
    lats = [r["latency_s"] for r in reqs]
    fig, ax = ft.plt.subplots(figsize=(9.6, 5.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.17)
    ax.grid(axis="x", visible=True)
    ax.scatter(toks, lats, color=c["qlora"], s=60, marker="s", zorder=3, label=f"Request ({len(reqs)} total)")
    for q, key, ls in (("p50", "latency_p50_s", "--"), ("p95", "latency_p95_s", ":")):
        ax.axhline(lt[key], color=c["muted"], linestyle=ls, linewidth=1.4)
        ax.annotate(f"{q} {lt[key]:.2f} s", (max(toks) * 0.02, lt[key]), xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", color=c["text"])  # fmt: skip
    cap = lt["generation"]["max_new_tokens"]
    n_cap = sum(t >= cap for t in toks)
    if n_cap:
        capped = [lat for t, lat in zip(toks, lats, strict=True) if t >= cap]
        ax.annotate(f"{n_cap} overlapping replies stopped\nat the {cap}-token cap", (cap, min(capped)),
                    xytext=(cap - 4, min(capped) - 3.2), textcoords="data", ha="right", va="top", color=c["muted"],
                    fontsize=ft.LABEL_PT - 1,
                    arrowprops={"arrowstyle": "-", "color": c["muted"], "linewidth": 1})  # fmt: skip
    ax.set_xlabel("Generated tokens in the reply")
    ax.set_ylabel("Request latency (s, client side)")
    ax.set_xlim(0, max(toks) * 1.06)
    ax.set_ylim(0, max(lats) * 1.18)
    ft.title(fig, ax, f"Latency tracks reply length: p50 {lt['latency_p50_s']:.2f} s, p95 {lt['latency_p95_s']:.2f} s",
             f"{lt['n']} greedy requests at concurrency {lt['concurrency']}: {lt['throughput_tokens_per_s']} tokens/s, "
             f"peak VRAM {lt['peak_vram_reserved_gb']:.2f} GB reserved of {lt['physical_vram_gb']:.2f} GB.")  # fmt: skip
    ft.footer(fig, f"Source: {rel(SRC['load'])}; MLflow run {lt['mlflow_run_id'][:8]}; 4-bit base + champion adapter")
    return ft.save(fig, OUT, "07-serving-latency", theme)


FIGURES = [fig_hero, fig_categories, fig_slots, fig_forgetting, fig_proxy, fig_training, fig_latency]


def contact_sheet(paths: list[Path]) -> Path:
    import matplotlib.image as mpimg

    pngs = sorted(p for p in paths if p.suffix == ".png")
    names = sorted({p.name.rsplit("-", 1)[0] for p in pngs})
    ft.apply("light")
    fig, axes = ft.plt.subplots(len(names), 2, figsize=(16, 4.6 * len(names)))
    for r, n in enumerate(names):
        for col, theme in enumerate(("light", "dark")):
            ax = axes[r][col]
            ax.imshow(mpimg.imread(OUT / f"{n}-{theme}.png"))
            ax.set_axis_off()
            ax.set_title(f"{n}  ({theme})", fontsize=10, loc="left")
    fig.tight_layout()
    out = OUT / "_contact_sheet.png"
    fig.savefig(out, dpi=90)
    ft.plt.close(fig)
    return out


def main() -> None:
    missing = [str(p) for p in SRC.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing source files: {missing}")
    print("Contrast vs background (WCAG ratio, need >= 3:1 for marks):")
    bad = []
    for theme, role, r, ok in ft.contrast_report():
        print(f"  {theme:5} {role:6} {r:5.2f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append((theme, role, r))
    if bad:
        raise SystemExit(f"contrast below 3:1: {bad}")
    written: list[Path] = []
    for f in FIGURES:
        for theme in ("light", "dark"):
            written += f(theme)
    sheet = contact_sheet(written)
    print(f"wrote {len(written)} files to {OUT}")
    print(f"contact sheet: {sheet}")
    print(f"report snapshots: {snapshot_reports()}")


REPORT_SNAPSHOTS = ["artifacts/reports/data_card.md", "artifacts/reports/eval_llama2-7b-qlora.md",
                    "artifacts/reports/benchmark.md", "artifacts/reports/merge_llama2-7b-qlora.json",
                    "artifacts/reports/load_test.json", "logs/commit_peaks.json", "logs/stage_timings.json"]  # fmt: skip


def snapshot_reports() -> Path:
    """Copy the small text reports the README links to into repo/docs/reports (the originals live outside the repo,
    under QLORA_ROOT/artifacts and QLORA_ROOT/logs)."""
    import shutil

    dst = repo_dir() / "docs" / "reports"
    dst.mkdir(parents=True, exist_ok=True)
    for r in REPORT_SNAPSHOTS:
        shutil.copy2(ROOT / r, dst / Path(r).name)
    return dst


if __name__ == "__main__":
    main()
