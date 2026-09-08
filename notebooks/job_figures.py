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

import job_analysis as ja
from job_analysis import (EARNED, LEVELS, OUTCOMES, REPO, STAGES,
                          cause_rows, fmt_tokens, token_quartile_table)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Ordinal ramps, light -> dark, one hue each.
BAND_COLOR = {"Easy": "#86b6ef", "Medium": "#256abf", "Hard": "#0d366b"}
QUARTILE_RAMP = ["#a8cbf2", "#5b9be0", "#256abf", "#0d366b"]
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

# Arms are not an ordered variable — no method is "more" than another — so they
# get distinct hues rather than a ramp. Assigned by position, so a comparison can
# label its arms whatever it likes without this module having to know the names.
# The reserved alert red is not among them.
ARM_RAMP = ["#9b8dc0", "#4c9f70", "#2a78d6", "#c2803f"]
# Outcome does run best to worst, so it gets two ramps meeting at the boundary
# that matters: green for the trials that earned something, the failure-stage
# blues for the rest, and grey for a fault that was never the method's doing.
EARNED_COLOR = {"solved": "#2f7d55", "partial credit": "#93c7ab"}

#: Figures are drawn without a headline by default: these are for a paper that
#: captions them in its own language, and an English sentence baked into the
#: image would have to be cropped out. Set this True to get them back — the
#: strings are all still here, and every one is a `text=` override away.
TITLES = False

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
    "token_quartiles": dict(
        title="The heaviest quarter of trials takes {share:.0%} of the tokens",
        x="", y="",
        panels=("mean reward", "share of the run's tokens"),
    ),
    "tokens_by_outcome": dict(
        title="{wasted} of {total} tokens bought no passing test",
        x="tokens", y="",
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
    # --- the cross-arm figures -------------------------------------------- #
    "reward_curves": dict(
        title="{leader} scores above zero on {n_leader} of {n} trials; {laggard} on {n_laggard}",
        x="trials, ranked by reward within each arm", y="reward",
    ),
    "reward_and_tokens": dict(
        title="{lo} to {hi} output tokens per solved task",
        x="", y="",
        panels=("mean reward", "output tokens"),
    ),
    "difficulty_slopes": dict(
        title="{leader} keeps {kept:.0%} of its Easy score on Hard, {laggard} {lost:.0%}",
        x="", y="mean reward",
    ),
    "outcome_mix": dict(
        title="What each arm's trials produced",
        x="share of trials", y="",
    ),
    "execution_gap": dict(
        title="{n} zeros are mistakes one execution would have caught",
        x="trials at zero", y="",
        note="none in the two arms that run what they write",
    ),
    "solved_overlap": dict(
        title="{union} tasks solved by some arm, {alone} of them by {leader} alone",
        x="tasks solved", y="",
        series=("only this arm", "also elsewhere"),
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
        shown = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        print(f"wrote {shown}")
        return path

    return save


def _text(key, override):
    """The strings for one figure: module defaults, then the caller's overrides.

    An explicit `text={"title": ...}` still draws a title with `TITLES` off, so a
    notebook can put one back on a single figure without changing the default.
    """
    strings = {**FIGURE_TEXT[key], **(override or {})}
    if not TITLES and "title" not in (override or {}):
        strings["title"] = ""
    return strings


def _title(target, string, **fields):
    """Set a title, unless it is empty — `target` is an axes or a figure."""
    if not string:
        return
    filled = string.format(**fields)
    if hasattr(target, "suptitle"):
        target.suptitle(filled, x=0.005, ha="left", fontsize=11,
                        fontweight="semibold", color=INK)
    else:
        target.set_title(filled)


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

    _title(ax, t["title"], n_zero=n_zero, n=len(ranked))
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

    _title(ax, t["title"], ratio=means["Easy"] / means["Hard"])
    ax.set_ylabel(t["y"])
    ax.set_ylim(0, 1.14)
    tidy(ax)
    return fig


def token_quartiles(df, *, text=None, figsize=(7.4, 3.3)):
    """What each quarter of the tokens returned, beside what it consumed.

    Two panels rather than a scatter of tokens against reward: the per-trial
    relationship is weak and a scatter overstates it, while the two aggregates
    are the shape of the budget question — reward falling left to right and
    consumption rising left to right is the whole argument.
    """
    t = _text("token_quartiles", text)
    group = token_quartile_table(df)
    ticks = list(group.index)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [(group.mean_reward, t["panels"][0], "{:.2f}"),
              (group.share, t["panels"][1], "{:.0%}")]
    for ax, (values, label, fmt) in zip(axes, panels):
        bars = ax.bar(range(len(values)), values, width=0.68, color=QUARTILE_RAMP, zorder=2)
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

    _title(fig, t["title"], share=group.share.iloc[-1])
    fig.tight_layout()
    return fig


def tokens_by_outcome(df, *, text=None, figsize=(6.4, 3.2)):
    """Where the tokens went, split by what they produced.

    Blue for the two outcomes that returned something, grey for the rest; the
    title totals the grey.
    """
    t = _text("tokens_by_outcome", text)
    present = [o for o in OUTCOMES if o in set(df.outcome)]
    used = df.groupby("outcome").total_tok.sum().reindex(present).fillna(0)
    count = df.outcome.value_counts().reindex(present).fillna(0).astype(int)
    wasted = used[[o for o in present if o not in EARNED]].sum()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(present[::-1], used[present[::-1]], height=0.66,
                   color=[SUBJECT if o in EARNED else CONTEXT for o in present[::-1]],
                   zorder=2)
    for rect, outcome in zip(bars, present[::-1]):
        ax.annotate(f"{fmt_tokens(used[outcome])}  ({count[outcome]})",
                    xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2)

    _title(ax, t["title"], wasted=fmt_tokens(wasted),
           total=fmt_tokens(df.total_tok.sum()))
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, used.max() * 1.32)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_tokens(v))
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
    _title(fig, t["title"], n=len(zeros), n_causes=zeros.fault_cause.nunique())
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# comparing arms
# --------------------------------------------------------------------------- #

def arm_colors(arms) -> list:
    """One hue per arm, assigned by position rather than by name."""
    return [ARM_RAMP[i % len(ARM_RAMP)] for i in range(len(arms))]


def outcome_color(outcome: str) -> str:
    """Green if the trial earned something, the stage ramp if it did not."""
    return {**EARNED_COLOR, **STAGE_COLOR, "lost to host fault": AXIS}[outcome]


def _on_color(background: str) -> str:
    """Ink or paper, whichever reads against that background."""
    r, g, b = (int(background[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else SURFACE


def _spread(values, gap):
    """Nudge label positions apart, keeping their order.

    Direct labels beat a legend right up until two of them land on the same y.
    This moves only what overlaps, and only far enough to clear `gap`.
    """
    placed = list(values)
    ascending = sorted(range(len(values)), key=lambda i: values[i])
    for lower, upper in zip(ascending, ascending[1:]):
        placed[upper] = max(placed[upper], placed[lower] + gap)
    return placed


def _label_arms(ax, arms, colors, at, positions, gap):
    """Write each arm's name on its own curve, pushed apart where they collide."""
    for arm, color, y in zip(arms, colors, _spread(positions, gap)):
        ax.annotate(arm, xy=(at, y), xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="semibold", va="center",
                    annotation_clip=False)


def reward_curves(df, *, text=None, figsize=(7.4, 3.6), at=0.78):
    """Every arm's trials sorted by reward, as one curve each.

    Four ranked distributions on one axis rather than four means. Where a curve
    leaves the floor is the share of the run that scored nothing, and how it
    climbs afterwards says whether the arm produces partial credit or finished
    packages — neither of which survives being averaged. `at` only decides where
    the labels sit.
    """
    t = _text("reward_curves", text)
    arms = ja.arm_order(df)
    colors = arm_colors(arms)

    fig, ax = plt.subplots(figsize=figsize)
    above, positions = {}, []
    for arm, color in zip(arms, colors):
        values = df.loc[(df.arm == arm) & df.scored, "reward"].sort_values().to_numpy()
        # Ranked as a share, so arms that scored a different number of trials
        # still line up and the zero block is read off the x-axis directly.
        x = np.linspace(0, 1, len(values))
        ax.plot(x, values, color=color, linewidth=2.1, zorder=3)
        above[arm] = (int((values > 0).sum()), len(values))
        positions.append(float(np.interp(at, x, values)))
    _label_arms(ax, arms, colors, at, positions, 0.085)

    leader = max(above, key=lambda a: above[a][0])
    laggard = min(above, key=lambda a: above[a][0])
    _title(ax, t["title"], leader=leader, n_leader=above[leader][0],
           n=above[leader][1], laggard=laggard, n_laggard=above[laggard][0])
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    tidy(ax)
    return fig


def reward_and_tokens(df, *, text=None, figsize=(7.6, 3.5)):
    """What each arm scored beside what it generated getting there.

    The bar is output tokens, not the total and not a bill. A bill depends on
    which model was priced and on how much input came from cache at a discount,
    and totals are dominated by an agent loop re-reading its own context — the
    four totals span 56x, which would leave the leanest arm invisible. Output
    tokens span 5x and are the model's actual production.

    Per-solved efficiency is a direct label rather than the bar for the same
    reason: those numbers span two orders of magnitude. The two denominators
    disagree about the ranking, and `ja.compare` carries both.
    """
    t = _text("reward_and_tokens", text)
    table = ja.compare(df)
    arms = list(table.index)
    colors = arm_colors(arms)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        (table.mean_reward, t["panels"][0],
         [f"{v:.2f}\n{n:.0f} solved" for v, n in zip(table.mean_reward, table.solved)]),
        (table.out_tok, t["panels"][1],
         [f"{fmt_tokens(v)}\n{fmt_tokens(p)}/solved"
          for v, p in zip(table.out_tok, table.out_per_solved)]),
    ]
    for ax, (values, label, marks) in zip(axes, panels):
        bars = ax.bar(range(len(values)), values, width=0.66, color=colors, zorder=2)
        for rect, value, mark in zip(bars, values, marks):
            ax.annotate(mark, xy=(rect.get_x() + rect.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK, linespacing=1.35)
        ax.set_ylabel(label)
        ax.set_ylim(0, values.max() * 1.34)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(arms, fontsize=8.5)
        ax.set_yticklabels([])
        tidy(ax)
        ax.grid(False)

    _title(fig, t["title"], lo=fmt_tokens(table.out_per_solved.min()),
           hi=fmt_tokens(table.out_per_solved.max()))
    fig.tight_layout()
    return fig


def difficulty_slopes(df, *, text=None, figsize=(6.0, 3.6)):
    """Mean reward across the three difficulty bands, one line per arm.

    The gradient is the point: two arms can share a mean and differ entirely in
    how much of it survives the hard half of the benchmark, which decides how a
    difficulty-skewed subset would rank them.
    """
    t = _text("difficulty_slopes", text)
    arms = ja.arm_order(df)
    colors = arm_colors(arms)
    means = (df[df.scored].pivot_table(index="difficulty", columns="arm",
                                       values="reward", aggfunc="mean", observed=True)
             .reindex(index=LEVELS, columns=arms))

    fig, ax = plt.subplots(figsize=figsize)
    for arm, color in zip(arms, colors):
        ax.plot(range(len(LEVELS)), means[arm], color=color, marker="o",
                markersize=5, linewidth=2, zorder=3)
    _label_arms(ax, arms, colors, len(LEVELS) - 1, list(means.loc["Hard"]), 0.055)

    kept = means.loc["Hard"] / means.loc["Easy"]
    _title(ax, t["title"], leader=kept.idxmax(), kept=kept.max(),
           laggard=kept.idxmin(), lost=kept.min())
    ax.set_ylabel(t["y"])
    ax.set_xlim(-0.15, len(LEVELS) - 0.35)
    ax.set_ylim(0, max(means.max()) * 1.12)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(LEVELS)
    tidy(ax)
    return fig


def outcome_mix(df, *, text=None, figsize=(7.6, 3.2)):
    """One bar per arm, split by what its trials produced.

    Reward as a single mean cannot say whether a run of 0.3 is many half-working
    packages or a few good ones among wreckage. This can.
    """
    t = _text("outcome_mix", text)
    table = ja.outcome_mix(df)
    table = table.loc[table.sum(axis=1) > 0]
    share = table / table.sum()
    arms = list(table.columns)[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(arms))
    for outcome in table.index:
        values = share.loc[outcome, arms].to_numpy()
        color = outcome_color(outcome)
        ax.barh(arms, values, left=left, height=0.62, color=color, zorder=2)
        for i, (value, start) in enumerate(zip(values, left)):
            if value >= 0.055:
                ax.annotate(f"{table.loc[outcome, arms[i]]}",
                            xy=(start + value / 2, i), ha="center", va="center",
                            fontsize=8.5, color=_on_color(color), zorder=3)
        left += values

    _title(ax, t["title"])
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=outcome_color(o))
                       for o in table.index],
              labels=list(table.index), loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, -0.28))
    tidy(ax, grid_axis="x")
    ax.grid(False)
    return fig


def execution_gap(df, *, text=None, figsize=(7.4, 3.2)):
    """Zeros caused by mistakes that running the code once would have exposed.

    A missing name, a file that does not parse, a package that cannot import
    itself: none of these needs the specification or the hidden tests to find.
    Splitting them by arm asks whether a method executes what it writes before
    handing it over, and the answer is visible as two empty rows.
    """
    t = _text("execution_gap", text)
    table = ja.cause_by_arm(df, ja.EXECUTION_CATCHABLE)
    table = table.loc[table.sum(axis=1) > 0]
    arms = list(table.columns)[::-1]
    totals = table.sum()

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(arms))
    for cause, color in zip(table.index, QUARTILE_RAMP):
        values = table.loc[cause, arms].to_numpy()
        ax.barh(arms, values, left=left, height=0.62, color=color, zorder=2)
        left += values
    for i, arm in enumerate(arms):
        ax.annotate(f"{totals[arm]}", xy=(totals[arm], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_2 if totals[arm] else SUBJECT,
                    fontweight="normal" if totals[arm] else "semibold")
    empty = [i for i, arm in enumerate(arms) if not totals[arm]]
    if empty:
        ax.annotate(t["note"], xy=(max(totals) * 0.06, sum(empty) / len(empty)),
                    color=SUBJECT, fontsize=9, va="center")

    _title(ax, t["title"], n=int(totals.sum()))
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, max(totals) * 1.18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c)
                       for c in QUARTILE_RAMP[:len(table)]],
              labels=list(table.index), loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.26))
    tidy(ax, grid_axis="x")
    return fig


def solved_overlap(df, *, text=None, figsize=(6.8, 2.9)):
    """Solved tasks per arm, split by whether any other arm solved them too.

    The dark part of each bar is what would be lost by dropping that arm. An arm
    whose bar is entirely light adds nothing to the set, whatever it scores on
    its own.
    """
    t = _text("solved_overlap", text)
    table = ja.solved_alone(df)
    solved = ja.paired(df, values="solved").astype(bool)
    arms = list(table.index)[::-1]
    colors = dict(zip(ja.arm_order(df), arm_colors(ja.arm_order(df))))
    shared_color = "#d5d4cd"

    fig, ax = plt.subplots(figsize=figsize)
    alone = table["only this arm"].reindex(arms)
    also = table["also solved elsewhere"].reindex(arms)
    ax.barh(arms, alone, height=0.62, color=[colors[a] for a in arms], zorder=2)
    ax.barh(arms, also, left=alone, height=0.62, color=shared_color, zorder=2)
    for i, arm in enumerate(arms):
        total = alone[arm] + also[arm]
        ax.annotate(f"{total}" + (f"   {alone[arm]} alone" if alone[arm] else ""),
                    xy=(total, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)

    # The dark segment is the arm's own hue and so cannot be put in a legend
    # without picking one arm's colour to stand for all four. It is labelled in
    # place instead, once, on the bar with room for it.
    widest = int(np.argmax(alone.to_numpy()))
    for label, start, width, ground in (
            (t["series"][0], 0, alone.iloc[widest], colors[arms[widest]]),
            (t["series"][1], alone.iloc[widest], also.iloc[widest], shared_color)):
        if width >= 0.20 * (alone + also).max():
            ax.annotate(label, xy=(start + width / 2, widest), ha="center",
                        va="center", fontsize=8.5, color=_on_color(ground), zorder=3)

    union = int(solved.any(axis=1).sum())
    _title(ax, t["title"], union=union, alone=int(table["only this arm"].max()),
           leader=table["only this arm"].idxmax())
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, (alone + also).max() * 1.28)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    tidy(ax, grid_axis="x")
    return fig
