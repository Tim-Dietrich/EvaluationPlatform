"""House style, and the five figures every run notebook draws the same way.

Each arm's notebook has two or three figures of its own — the thing that only
that method can be asked about. Everything else is common, and is here so that
the same question is drawn the same way for every arm. A reader comparing two
notebooks should be comparing the runs, not two people's axis choices.

Conventions
-----------

**Ordered variables get ordered ramps.** Difficulty, cost quartile and
failure-stage are all ordered, so each gets one hue running light to dark rather
than a set of arbitrary colours. Where none of them is the subject, grey is
context and blue is the thing being shown; red is reserved for the one thing a
figure is warning about.

**Text on a figure is a title, an axis word, and direct labels on the marks.**
Every string a shared figure can draw lives in `FIGURE_TEXT`, and every figure
takes a `text=` override, so re-wording a chart never means touching plotting
code:

    fig = jf.reward_ranked(df, text={"title": "{n_zero} runs came back empty"})

Titles that quote a number are format strings filled from the data at draw time,
so they cannot go stale when the job changes.

Using it
--------

    import job_figures as jf

    jf.use_house_style()
    save = jf.saver(prefix="terminus")

    save(jf.reward_ranked(df), "reward-ranked")
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from job_analysis import (EARNED, LEVELS, OUTCOMES, REPO, STAGES,
                          cause_rows, cost_quartile_table)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Ordinal ramps, light -> dark, one hue each.
BAND_COLOR = {"Easy": "#86b6ef", "Medium": "#256abf", "Hard": "#0d366b"}
COST_RAMP = ["#a8cbf2", "#5b9be0", "#256abf", "#0d366b"]
# How far a failed run got: dark is a run that stopped earliest. The harness
# fault is grey because it is not a point on that pipeline at all.
STAGE_COLOR = {
    "never configured": "#0d366b",
    "never imported": "#256abf",
    "ran, all failed": "#86b6ef",
    "no verdict": "#898781",
}
# Where none of those orderings is in play: grey is context, blue is the subject.
CONTEXT = "#898781"
SUBJECT = "#2a78d6"
# Reserved for the one thing a figure is warning about.
ALERT = "#b4472e"

FIGURES = REPO / "docs" / "figures"

#: Every string the shared figures can draw. `{...}` fields are filled from the
#: data. A literal dollar sign is written `\$`: matplotlib reads a bare pair of
#: them as maths.
FIGURE_TEXT = {
    "reward_ranked": dict(
        title="{n_zero} of {n} scored trials score zero",
        x="scored trials, ranked by reward", y="reward",
        note="{n_zero} at zero", rug="difficulty",
    ),
    "reward_by_difficulty": dict(
        title="Reward falls {ratio:.0f}x from Easy to Hard",
        x="", y="mean reward",
    ),
    "cost_quartiles": dict(
        title="The dearest quarter of trials takes {share:.0%} of the spend",
        x="", y="",
        panels=("mean reward", "share of the run's spend"),
    ),
    "spend_by_outcome": dict(
        title=r"\${wasted:.2f} of \${total:.2f} bought no passing test",
        x="spend (USD)", y="",
    ),
    "zero_causes": dict(
        title="{n} zeros, {n_causes} distinct causes",
        x="trials", y="",
        series={
            "never configured": "pytest never started",
            "never imported": "the code never imported",
            "ran, all failed": "tests ran, all failed",
            "no verdict": "no verdict (harness)",
        },
    ),
}

RC_PARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2,
}


def use_house_style() -> None:
    """Apply the shared rcParams. Call once, near the top of a notebook."""
    mpl.rcParams.update(RC_PARAMS)


def tidy(ax, *, grid_axis="y"):
    """Recessive chrome: no box, one axis of hairline grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis=grid_axis, which="major")
    ax.grid(axis={"y": "x", "x": "y"}[grid_axis], which="major", visible=False)
    ax.tick_params(length=0)
    return ax


def saver(prefix: str, directory=None):
    """A `save(fig, name)` that writes `<prefix>-<name>.png` and says where."""
    figures = directory or FIGURES
    figures.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        path = figures / f"{prefix}-{name}.png"
        fig.savefig(path)
        print(f"wrote {path.relative_to(REPO)}")
        return path

    return save


def _text(key, override):
    """The strings for one figure: module defaults, then the caller's overrides."""
    return {**FIGURE_TEXT[key], **(override or {})}


def difficulty_legend(ax, **kwargs):
    """One swatch per level, for a figure whose colour means difficulty."""
    return ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR[d]) for d in LEVELS],
        labels=LEVELS, **{"loc": "upper left", "ncol": 3, **kwargs})


# --------------------------------------------------------------------------- #
# the shared five
# --------------------------------------------------------------------------- #

def reward_ranked(df, *, text=None, figsize=(7.2, 3.3)):
    """Every scored trial, ranked, with the zero block marked and measured.

    Trials that never reached a verifier are not drawn: they have no reward to
    rank. Ties break by difficulty, so the zero block reads as three measurable
    segments rather than as noise — and because bars of zero height show no
    colour at all, a rug under the axis carries difficulty for every trial alike.
    """
    t = _text("reward_ranked", text)
    ranked = df[df.scored].sort_values(["reward", "difficulty"]).reset_index(drop=True)
    n_zero = int((ranked.reward == 0).sum())
    colors = [BAND_COLOR[d] for d in ranked.difficulty]
    rug_top, rug_bottom = -0.015, -0.075

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(ranked.index, ranked.reward, width=1.0, color=colors, linewidth=0)
    ax.bar(ranked.index, rug_top - rug_bottom, width=1.0, bottom=rug_bottom,
           color=colors, linewidth=0)
    ax.axvline(n_zero - 0.5, color=ALERT, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
    ax.annotate(t["note"].format(n_zero=n_zero), xy=(n_zero - 3, 0.10),
                color=ALERT, ha="right", fontsize=9)
    ax.annotate(t["rug"], xy=(len(ranked) + 1, (rug_top + rug_bottom) / 2),
                color=MUTED, va="center", fontsize=8, annotation_clip=False)

    ax.set_title(t["title"].format(n_zero=n_zero, n=len(ranked)))
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.set_xlim(-1, len(ranked))
    ax.set_ylim(rug_bottom, 1.05)
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0, 1.0)
    difficulty_legend(ax)
    tidy(ax)
    return fig


def reward_by_difficulty(df, *, text=None, figsize=(5.4, 3.4), seed=0):
    """Mean reward per level, with every trial drawn over it.

    The spread matters as much as the mean here, so the points are not optional
    decoration: a bar alone would imply a precision these distributions do not
    have. `seed` fixes the jitter; no reported number depends on it.
    """
    t = _text("reward_by_difficulty", text)
    scored = df[df.scored]
    means = scored.groupby("difficulty", observed=True).reward.mean().reindex(LEVELS)
    counts = scored.groupby("difficulty", observed=True).reward.size().reindex(LEVELS)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(LEVELS, means, width=0.62, color=[BAND_COLOR[d] for d in LEVELS], zorder=2)
    for i, level in enumerate(LEVELS):
        points = scored.loc[scored.difficulty == level, "reward"]
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(points)), points,
                   s=11, color=INK, alpha=0.35, linewidth=0, zorder=3)
        ax.annotate(f"{means[level]:.2f}\nn={counts[level]}", xy=(i, means[level]),
                    xytext=(0, 5), textcoords="offset points", ha="center", color=INK,
                    fontsize=9, fontweight="semibold", linespacing=1.3)

    ax.set_title(t["title"].format(ratio=means["Easy"] / means["Hard"]))
    ax.set_ylabel(t["y"])
    ax.set_ylim(0, 1.14)
    tidy(ax)
    return fig


def cost_quartiles(df, *, text=None, figsize=(7.4, 3.3)):
    """What each quarter of the spend returned, beside what it consumed.

    Two panels rather than a scatter of cost against reward: the per-trial
    relationship is weak and a scatter overstates it, while the two aggregates
    are the shape of the budget question — reward falling left to right and
    spend rising left to right is the whole argument.
    """
    t = _text("cost_quartiles", text)
    group = cost_quartile_table(df)
    ticks = list(group.index)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [(group.mean_reward, t["panels"][0], "{:.2f}"),
              (group.share, t["panels"][1], "{:.0%}")]
    for ax, (values, label, fmt) in zip(axes, panels):
        bars = ax.bar(range(len(values)), values, width=0.68, color=COST_RAMP, zorder=2)
        for rect, value in zip(bars, values):
            ax.annotate(fmt.format(value),
                        xy=(rect.get_x() + rect.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=9.5, color=INK, fontweight="semibold")
        ax.set_ylabel(label)
        ax.set_ylim(0, values.max() * 1.25)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(ticks, fontsize=8.5, linespacing=1.4)
        ax.set_yticklabels([])
        tidy(ax)
        ax.grid(False)

    fig.suptitle(t["title"].format(share=group.share.iloc[-1]),
                 x=0.005, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    return fig


def spend_by_outcome(df, *, text=None, figsize=(6.4, 3.2)):
    """Where the money went, split by what it produced.

    Blue for the two outcomes that returned something, grey for the rest; the
    title totals the grey.
    """
    t = _text("spend_by_outcome", text)
    present = [o for o in OUTCOMES if o in set(df.outcome)]
    spend = df.groupby("outcome").cost_usd.sum().reindex(present).fillna(0)
    count = df.outcome.value_counts().reindex(present).fillna(0).astype(int)
    wasted = spend[[o for o in present if o not in EARNED]].sum()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(present[::-1], spend[present[::-1]], height=0.66,
                   color=[SUBJECT if o in EARNED else CONTEXT for o in present[::-1]],
                   zorder=2)
    for rect, outcome in zip(bars, present[::-1]):
        ax.annotate(f"${spend[outcome]:.2f}  ({count[outcome]})",
                    xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2)

    ax.set_title(t["title"].format(wasted=wasted, total=df.cost_usd.sum()))
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, spend.max() * 1.32)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:g}")
    tidy(ax, grid_axis="x")
    return fig


def zero_causes(df, *, text=None, width=7.6):
    """Every zero by its technical cause, grouped and coloured by stage.

    The rows are as tall as the taxonomy needs, so the figure height is derived
    rather than fixed. The same exception can appear under two stages — a
    `SyntaxError` that stopped an import is a different finding from one a test
    hit — and the stage colour is what tells them apart.
    """
    t = _text("zero_causes", text)
    zeros = df[df.scored & (df.reward == 0)]
    rows = cause_rows(df)
    labels = [cause for cause, _, _ in rows][::-1]
    values = [n for _, n, _ in rows][::-1]
    colors = [STAGE_COLOR[stage] for _, _, stage in rows][::-1]

    fig, ax = plt.subplots(figsize=(width, 0.32 * len(rows) + 1.4))
    bars = ax.barh(range(len(rows)), values, height=0.7, color=colors, zorder=2)
    for rect, value in zip(bars, values):
        ax.annotate(str(value), xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2)

    ax.set_xlabel(t["x"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    shown = [s for s in STAGES if any(r[2] == s for r in rows)]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOR[s]) for s in shown],
              labels=[t["series"][s] for s in shown], loc="lower right", ncol=1)
    tidy(ax, grid_axis="x")
    # The cause labels are long, so the axes starts well to the right; titling
    # the figure rather than the axes keeps the headline against the left margin.
    fig.suptitle(t["title"].format(n=len(zeros), n_causes=zeros.fault_cause.nunique()),
                 x=0.005, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    return fig
