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

import re

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

#: Figures are drawn in German. The analysis is not, and must not be: causes,
#: stages, outcomes and endings are dictionary keys, column values and index
#: labels throughout `job_analysis`, and translating those would mean
#: translating the classifier and every lookup built on it. Only what a figure
#: paints goes through `de`, so the printed tables stay in the language the code
#: is written in and the pictures come out in the language of the paper.
LANGUAGE = "de"

#: Exception names are deliberately left alone. `SyntaxError` is what the
#: interpreter prints and what a reader would search for; only the description
#: this analysis appends after the colon is ours to translate, so `de` also
#: matches on that tail alone and each description is written here once however
#: many exceptions carry it.
GERMAN = {
    # difficulty, as the benchmark labels it
    "Easy": "Einfach", "Medium": "Mittel", "Hard": "Schwer",
    # what a trial produced
    "solved": "gelöst",
    "partial credit": "Teilpunkte",
    "lost to host fault": "Host-Fehler",
    # how far it got before it stopped
    "never configured": "nie konfiguriert",
    "never imported": "nie importiert",
    "ran, all failed": "gelaufen, alle fehlgeschlagen",
    "no verdict": "kein Urteil",
    # the descriptions appended to an exception name
    "own code, wrong layout": "eigener Code, falsches Layout",
    "a file it never wrote": "nie geschriebene Datei",
    "uninstalled dependency": "nicht installierte Abhängigkeit",
    "project built one level too deep": "Projekt eine Ebene zu tief angelegt",
    "circular import": "zirkulärer Import",
    "name gone from an installed module": "Name im installierten Modul entfallen",
    "name missing from its own module": "Name fehlt im eigenen Modul",
    # causes that are not an exception at all
    "tester timed out": "Zeitüberschreitung des Testers",
    "pytest missing from the image": "pytest fehlt im Image",
    "pyproject.toml is not valid TOML": "pyproject.toml ist kein gültiges TOML",
    "setup.cfg uses a removed pytest section":
        "setup.cfg nutzt einen entfernten pytest-Abschnitt",
    "pytest config needs an uninstalled plugin":
        "pytest-Konfiguration braucht ein nicht installiertes Plugin",
    "no tests were collected": "keine Tests gesammelt",
    "test run killed mid-suite": "Testlauf mittendrin abgebrochen",
    "unclassified": "nicht klassifiziert",
    # how a run ended, per arm
    "reply finished": "Antwort abgeschlossen",
    "cut at the output ceiling": "am Ausgabelimit abgeschnitten",
    "declared complete": "als fertig gemeldet",
    "cut at turn ceiling": "am Turn-Limit abgeschnitten",
    "used every round, never self-verified":
        "alle Runden genutzt, nie selbst verifiziert",
    "finished the whole pipeline": "Pipeline vollständig durchlaufen",
    "stopped at the token budget": "am Token-Budget gestoppt",
    "stopped at its own wall clock": "an der eigenen Laufzeitgrenze gestoppt",
    "stopped at the harness deadline": "an der Harness-Frist abgebrochen",
    "killed by a signal": "vom System abgebrochen (Exit 137)",
    "ended from outside": "von außen beendet",
    "wrote nothing at all": "gar nichts geschrieben",
    # quartile ticks, which carry their own line break
    "Q1\nleanest": "Q1\nschlankste", "Q4\nheaviest": "Q4\nschwerste",
}

#: Labels that carry a number and so cannot be looked up whole.
GERMAN_PATTERNS = (
    (re.compile(r"^passed its own tests at round (\d+)$"),
     r"eigene Tests in Runde \1 bestanden"),
)


def num(value, spec: str = ".2f") -> str:
    """A number written the way the figure language writes it.

    German uses a decimal comma, and a figure printing `0.53` beside a table
    printing `0,53` is the kind of detail a reader notices and an author has to
    fix by hand. `LANGUAGE` governs both.
    """
    return format(value, spec).replace(".", ",") if LANGUAGE == "de" else format(value, spec)


def percent(value, decimals: int = 0) -> str:
    """A share as a percentage, with the language's decimal mark and spacing."""
    text = num(value * 100, f".{decimals}f")
    return text + (" %" if LANGUAGE == "de" else "%")


def tokens(value) -> str:
    """`fmt_tokens`, with the figure language's decimal mark."""
    text = fmt_tokens(value)
    return text.replace(".", ",") if LANGUAGE == "de" else text


def decimals(axis, spec: str = ".1f") -> None:
    """Write a numeric axis in the figure language, comma included."""
    axis.set_major_formatter(lambda value, _: num(value, spec))


def de(label) -> str:
    """The German display form of one label, or the label unchanged."""
    text = str(label)
    if LANGUAGE != "de":
        return text
    if text in GERMAN:
        return GERMAN[text]
    for pattern, replacement in GERMAN_PATTERNS:
        if pattern.match(text):
            return pattern.sub(replacement, text)
    # "ExceptionName: our description" — keep the name, translate the tail.
    name, sep, description = text.partition(": ")
    if sep and description in GERMAN:
        return f"{name}: {GERMAN[description]}"
    return text


FIGURES = REPO / "docs" / "figures"

#: Every string the shared figures can draw. `{...}` fields are filled from the
#: data. A literal dollar sign is written `\$`: matplotlib reads a bare pair of
#: them as maths.
FIGURE_TEXT = {
    "reward_ranked": dict(
        title="{n_zero} von {n} gewerteten Läufen erreichen null",
        x="gewertete Läufe, nach Reward sortiert", y="Reward",
        note="{n_zero} bei null", rug="Schwierigkeit",
    ),
    "reward_by_difficulty": dict(
        title="Reward fällt um das {ratio:.0f}-fache von Einfach zu Schwer",
        x="", y="mittlerer Reward",
    ),
    "token_quartiles": dict(
        title="Das schwerste Viertel verbraucht {share:.0%} der Tokens",
        x="", y="",
        panels=("mittlerer Reward", "Anteil an den Tokens des Laufs"),
    ),
    "tokens_by_outcome": dict(
        title="{wasted} von {total} Tokens ohne einen bestandenen Test",
        x="Tokens", y="",
    ),
    "zero_causes": dict(
        title="{n} Nullwertungen, {n_causes} verschiedene Ursachen",
        x="Läufe", y="",
        series={
            "never configured": "pytest nie gestartet",
            "never imported": "Code nie importiert",
            "ran, all failed": "Tests gelaufen, alle fehlgeschlagen",
            "no verdict": "kein Urteil (Harness)",
        },
    ),
    # --- the cross-arm figures -------------------------------------------- #
    "reward_curves": dict(
        title="{leader} liegt bei {n_leader} von {n} Läufen über null, {laggard} bei {n_laggard}",
        x="Läufe, je Ansatz nach Reward sortiert", y="Reward",
    ),
    "reward_and_tokens": dict(
        title="{lo} bis {hi} Ausgabe-Tokens je gelöster Aufgabe",
        x="", y="",
        panels=("mittlerer Reward", "Tokens insgesamt"),
        solved="{n:.0f} gelöst",
        total="{tokens}",
    ),
    "token_composition": dict(
        title="Woraus sich der Tokenverbrauch zusammensetzt",
        x="Anteil am Verbrauch des Ansatzes", y="",
        # The platform's own column names, kept in English: these are the terms
        # the run summary shows, and the German renderings are not established
        # vocabulary, so translating them would only make the figure harder to
        # check against the source.
        series=("Uncached Input Tokens", "Cached Input Tokens", "Output Tokens"),
        total="{tokens} gesamt",
    ),
    "difficulty_slopes": dict(
        title="{leader} hält {kept:.0%} des Einfach-Werts auf Schwer, {laggard} {lost:.0%}",
        x="", y="mittlerer Reward",
    ),
    "difficulty_spread": dict(
        title="Nicht nur das Mittel unterscheidet sich, sondern die Streuung",
        x="", y="Reward",
        note="Kasten: Quartile und Median   ×: Mittelwert",
    ),
    "runtime_spread": dict(
        title="Wie lange ein Lauf dauert, je Ansatz",
        x="Laufzeit des Agenten (min)", y="",
        row="Median {median} min   ·   {hours} h insgesamt",
        note="Kasten: Quartile und Median",
        limit="Harness-Frist\n{minutes} min",
    ),
    "outcome_mix": dict(
        title="Was die Läufe je Ansatz hervorgebracht haben",
        x="Anteil der Läufe", y="",
    ),
    "cause_overview": dict(
        title="{n} Nullwertungen über alle Ansätze, {n_causes} verschiedene Ursachen",
        x="Läufe bei null", y="",
    ),
    "execution_gap": dict(
        title="{n} Nullwertungen wären bei einer einzigen Ausführung aufgefallen",
        x="Läufe bei null", y="",
        note="keine bei den beiden Ansätzen,\ndie ihren Code ausführen",
    ),
    "solved_overlap": dict(
        title="{union} Aufgaben von irgendeinem Ansatz gelöst, {alone} nur von {leader}",
        x="gelöste Aufgaben", y="",
        series=("nur dieser Ansatz", "auch andere"),
        note="{n} exklusiv",
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

    The title is kept here whether or not it will be drawn — `TITLES` decides
    that, in `title`, so that a figure a notebook draws for itself out of its own
    dictionary of strings obeys the same switch as the shared ones.
    """
    return {**FIGURE_TEXT[key], **(override or {})}


def title(target, string, **fields):
    """Set a title, unless titles are off or the string is empty.

    `target` is an axes or a figure. Every figure in the project goes through
    here, including the ones a notebook builds itself, so `TITLES` is the single
    place that decides whether a headline is painted into the image.
    """
    if not TITLES or not string:
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
        labels=[de(level) for level in LEVELS],
        **{"loc": "upper left", "ncol": 3, **kwargs})


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

    title(ax, t["title"], n_zero=n_zero, n=len(ranked))
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.set_xlim(-1, len(ranked))
    ax.set_ylim(rug_bottom, 1.05)
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    decimals(ax.yaxis, ".1f")
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
    ax.bar([de(d) for d in LEVELS], means, width=0.62,
           color=[BAND_COLOR[d] for d in LEVELS], zorder=2)
    for i, level in enumerate(LEVELS):
        points = scored.loc[scored.difficulty == level, "reward"]
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(points)), points,
                   s=11, color=INK, alpha=0.35, linewidth=0, zorder=3)
        ax.annotate(f"{num(means[level])}\nn={counts[level]}", xy=(i, means[level]),
                    xytext=(0, 5), textcoords="offset points", ha="center", color=INK,
                    fontsize=9, fontweight="semibold", linespacing=1.3)

    title(ax, t["title"], ratio=means["Easy"] / means["Hard"])
    ax.set_ylabel(t["y"])
    ax.set_ylim(0, 1.14)
    decimals(ax.yaxis, ".1f")
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
    panels = [(group.mean_reward, t["panels"][0], lambda v: num(v)),
              (group.share, t["panels"][1], percent)]
    for ax, (values, label, fmt) in zip(axes, panels):
        bars = ax.bar(range(len(values)), values, width=0.68, color=QUARTILE_RAMP, zorder=2)
        for rect, value in zip(bars, values):
            ax.annotate(fmt(value),
                        xy=(rect.get_x() + rect.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=9.5, color=INK, fontweight="semibold")
        ax.set_ylabel(label)
        ax.set_ylim(0, values.max() * 1.25)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([de(tick) for tick in ticks], fontsize=8.5,
                           linespacing=1.4)
        ax.set_yticklabels([])
        tidy(ax)
        ax.grid(False)

    title(fig, t["title"], share=group.share.iloc[-1])
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
    bars = ax.barh([de(o) for o in present[::-1]], used[present[::-1]], height=0.66,
                   color=[SUBJECT if o in EARNED else CONTEXT for o in present[::-1]],
                   zorder=2)
    for rect, outcome in zip(bars, present[::-1]):
        ax.annotate(f"{tokens(used[outcome])}  ({count[outcome]})",
                    xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2)

    title(ax, t["title"], wasted=fmt_tokens(wasted),
           total=fmt_tokens(df.total_tok.sum()))
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, used.max() * 1.32)
    ax.xaxis.set_major_formatter(lambda v, _: tokens(v))
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
    ax.set_yticklabels([de(label) for label in labels], fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    shown = [s for s in STAGES if any(r[2] == s for r in rows)]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=STAGE_COLOR[s]) for s in shown],
              labels=[t["series"][s] for s in shown], loc="lower right", ncol=1)
    tidy(ax, grid_axis="x")
    # The cause labels are long, so the axes starts well to the right; titling
    # the figure rather than the axes keeps the headline against the left margin.
    title(fig, t["title"], n=len(zeros), n_causes=zeros.fault_cause.nunique())
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
    title(ax, t["title"], leader=leader, n_leader=above[leader][0],
           n=above[leader][1], laggard=laggard, n_laggard=above[laggard][0])
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.05)
    decimals(ax.yaxis, ".1f")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _: percent(v))
    tidy(ax)
    return fig


def reward_and_tokens(df, *, text=None, figsize=(7.4, 3.4)):
    """What each arm scored, beside how much it consumed getting there.

    Two panels, one number each. The right-hand bar is the total — everything
    read and written, which is the figure the run platform reports — because a
    comparison that quoted only output would understate an agent loop that reads
    twenty times what it writes, and would then disagree with the platform's own
    summary for no visible reason.

    What that total is *made of*, and what it bought, are separate questions and
    get their own figure in `token_composition`. Earlier versions of this one
    tried to carry all three at once and were unreadable.
    """
    t = _text("reward_and_tokens", text)
    table = ja.compare(df)
    arms = list(table.index)
    colors = arm_colors(arms)
    spots = range(len(arms))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        (table.mean_reward, t["panels"][0],
         [num(v) + "\n" + t["solved"].format(n=n)
          for v, n in zip(table.mean_reward, table.solved)]),
        (table.total_tok, t["panels"][1],
         [t["total"].format(tokens=tokens(v)) for v in table.total_tok]),
    ]
    for ax, (values, label, marks) in zip(axes, panels):
        bars = ax.bar(spots, values, width=0.66, color=colors, zorder=2)
        for rect, value, mark in zip(bars, values, marks):
            ax.annotate(mark, xy=(rect.get_x() + rect.get_width() / 2, value),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK, linespacing=1.35)
        ax.set_ylabel(label)
        ax.set_ylim(0, values.max() * 1.3)
        ax.set_xticks(spots)
        ax.set_xticklabels(arms, fontsize=8.5)
        ax.set_yticklabels([])
        tidy(ax)
        ax.grid(False)

    title(fig, t["title"], lo=fmt_tokens(table.out_per_solved.min()),
          hi=fmt_tokens(table.out_per_solved.max()))
    fig.tight_layout()
    return fig


def token_composition(df, *, text=None, figsize=(7.8, 2.9)):
    """What each arm's consumption is made of, as a share of its own total.

    Normalised per arm rather than drawn to a common scale, because the totals
    span 56x and on one axis the leanest arm's whole bar is thinner than the
    thickest arm's rounding. The absolute total sits at the end of each row, so
    the scale is not lost — it is just not what this figure is asking.

    The mix is a real finding rather than bookkeeping. One arm spends most of
    its budget generating; two spend almost all of it re-reading a transcript
    that is mostly served from cache. That difference is invisible in a total
    and it is what makes a cheap-looking arm expensive to run, or the reverse.
    """
    t = _text("token_composition", text)
    table = ja.compare(df)
    arms = list(table.index)[::-1]
    parts = [("uncached_tok", QUARTILE_RAMP[1]),
             ("cache_tok", QUARTILE_RAMP[0]),
             ("out_tok", QUARTILE_RAMP[3])]

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(arms))
    for (column, color), label in zip(parts, t["series"]):
        share = (table.loc[arms, column] / table.loc[arms, "total_tok"]).to_numpy()
        ax.barh(arms, share, left=left, height=0.62, color=color, zorder=2,
                label=label)
        for row, (value, start) in enumerate(zip(share, left)):
            if value >= 0.07:
                ax.annotate(percent(value), xy=(start + value / 2, row),
                            ha="center", va="center", fontsize=8.5,
                            color=_on_color(color), zorder=3)
        left += share

    # The absolute total only, so the row end says what the scale is without
    # also arguing about efficiency — that is a different question, and the
    # per-solved figures are in `ja.compare` and in the summary table.
    for row, arm in enumerate(arms):
        ax.annotate(t["total"].format(tokens=tokens(table.total_tok[arm])),
                    xy=(1.0, row), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)

    ax.set_xlabel(t["x"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: percent(v))
    ax.legend(loc="upper center", ncol=len(parts), bbox_to_anchor=(0.5, -0.26))
    tidy(ax, grid_axis="x")
    ax.grid(False)
    title(ax, t["title"])
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
    title(ax, t["title"], leader=kept.idxmax(), kept=kept.max(),
           laggard=kept.idxmin(), lost=kept.min())
    ax.set_ylabel(t["y"])
    ax.set_xlim(-0.15, len(LEVELS) - 0.35)
    ax.set_ylim(0, max(means.max()) * 1.12)
    decimals(ax.yaxis, ".1f")
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([de(level) for level in LEVELS])
    tidy(ax)
    return fig


def runtime_spread(df, *, text=None, figsize=(7.8, 3.8), seed=0):
    """How long a run takes, per arm: quartiles, median, and every trial.

    A box alone would hide that these distributions are not one shape. One arm
    is fast on most tasks and strung out on the rest; another sits in a narrow
    band whatever it is given; a third piles up against a deadline. A strip of
    the trials over the box shows which of those is happening, and the deadline
    is drawn wherever the data says one was reached.

    Points carry difficulty, so the figure also answers whether an arm's runtime
    responds to the size of the task at all — for two of these four it does not.

    Wall clock is not a compute cost and must not be read as one: an arm running
    forty requests at a time finishes sooner per unit of work than a sequential
    one, and `ja.compare` carries the token counts for that question.
    """
    t = _text("runtime_spread", text)
    arms = ja.arm_order(df)[::-1]
    minutes = df.t_agent / 60
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize)
    for row, arm in enumerate(arms):
        part = df[df.arm == arm]
        values = (part.t_agent / 60).dropna()
        if not len(values):
            continue
        ax.scatter(values, row + rng.uniform(-0.2, 0.2, len(values)),
                   s=16, color=[BAND_COLOR[d] for d in part.loc[values.index, "difficulty"]],
                   alpha=0.75, linewidth=0, zorder=3)
        box = ax.boxplot(values, positions=[row], widths=0.5, vert=False,
                         showfliers=False, showmeans=False,
                         medianprops=dict(color=INK, linewidth=1.8),
                         boxprops=dict(color=INK_2, linewidth=1.2),
                         whiskerprops=dict(color=INK_2, linewidth=1.1),
                         capprops=dict(color=INK_2, linewidth=1.1))
        for patch in box["boxes"]:
            patch.set_zorder(4)
        ax.annotate(t["row"].format(median=num(values.median(), ".1f"),
                                    hours=num(part.t_agent.sum() / 3600, ".1f")),
                    xy=(0, row + 0.30), color=INK_2, fontsize=8.5, va="bottom")

    # Only some runs meet a deadline, and only where one was actually recorded.
    seen = df["harness_deadline"].dropna() if "harness_deadline" in df else []
    if len(seen):
        limit = max(seen) / 60
        ax.axvline(limit, color=ALERT, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        ax.annotate(t["limit"].format(minutes=num(limit, ".0f")),
                    xy=(limit, len(arms) - 0.45), xytext=(-5, 0),
                    textcoords="offset points", ha="right", va="top",
                    color=ALERT, fontsize=8, linespacing=1.3)

    title(ax, t["title"])
    ax.set_xlabel(t["x"])
    ax.set_xlim(-1.5, minutes.max() * 1.04)
    ax.set_ylim(-0.6, len(arms) - 0.05)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=9)
    ax.annotate(t["note"], xy=(1.0, 1.0), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=7.5, color=MUTED)
    tidy(ax, grid_axis="x")
    difficulty_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.17))
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

    title(ax, t["title"])
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: percent(v))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=outcome_color(o))
                       for o in table.index],
              labels=[de(outcome) for outcome in table.index],
              loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.28))
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

    title(ax, t["title"], n=int(totals.sum()))
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, max(totals) * 1.18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c)
                       for c in QUARTILE_RAMP[:len(table)]],
              labels=[de(cause) for cause in table.index],
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.26))
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
        extra = "   " + t["note"].format(n=alone[arm]) if alone[arm] else ""
        ax.annotate(f"{total}" + extra,
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
    title(ax, t["title"], union=union, alone=int(table["only this arm"].max()),
           leader=table["only this arm"].idxmax())
    ax.set_xlabel(t["x"])
    ax.set_xlim(0, (alone + also).max() * 1.28)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    tidy(ax, grid_axis="x")
    return fig


def difficulty_spread(df, *, text=None, figsize=(8.2, 4.0), seed=0):
    """Every arm's reward distribution within every difficulty band.

    A mean per band says which arm is ahead and nothing about how. These
    distributions are zero-inflated and several of them are bimodal — a block of
    finished packages and a block of wreckage with little between — so the mean
    sits in a region where the arm has almost no trials at all.

    Three groups of four boxes: quartiles and median in the box, whiskers to the
    furthest point within 1.5 IQR, the mean as a cross, and every trial drawn
    behind as a jittered point. `seed` fixes the jitter and no reported number
    depends on it.
    """
    t = _text("difficulty_spread", text)
    arms = ja.arm_order(df)
    colors = arm_colors(arms)
    scored = df[df.scored]
    rng = np.random.default_rng(seed)

    # Four boxes per band, a whole band's width apart, with a gap between bands.
    step, span = 1.0, len(arms) + 1.0
    fig, ax = plt.subplots(figsize=figsize)
    for band, level in enumerate(LEVELS):
        for slot, (arm, color) in enumerate(zip(arms, colors)):
            values = scored.loc[(scored.arm == arm) & (scored.difficulty == level),
                                "reward"].to_numpy()
            if not len(values):
                continue
            at = band * span + slot * step
            ax.scatter(at + rng.uniform(-0.28, 0.28, len(values)), values,
                       s=9, color=color, alpha=0.4, linewidth=0, zorder=2)
            box = ax.boxplot(
                values, positions=[at], widths=0.62, showfliers=False,
                showmeans=True, meanprops=dict(marker="x", markersize=5,
                                               markeredgecolor=INK,
                                               markeredgewidth=1.2),
                medianprops=dict(color=INK, linewidth=1.6),
                boxprops=dict(color=color, linewidth=1.4),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2))
            for patch in box["boxes"]:
                patch.set_zorder(3)

    centre = (len(arms) - 1) / 2
    ax.set_xticks([band * span + centre for band in range(len(LEVELS))])
    ax.set_xticklabels([de(level) for level in LEVELS])
    ax.set_xlim(-0.8, (len(LEVELS) - 1) * span + len(arms) - 0.2)
    ax.set_ylim(-0.04, 1.06)
    decimals(ax.yaxis, ".1f")
    ax.set_ylabel(t["y"])
    title(ax, t["title"])
    # One legend, because twelve boxes cannot each carry a direct label.
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
              labels=arms, loc="upper center", ncol=len(arms),
              bbox_to_anchor=(0.5, -0.13))
    ax.annotate(t["note"], xy=(1.0, 1.02), xycoords="axes fraction", ha="right",
                fontsize=8, color=MUTED)
    tidy(ax)
    return fig


def cause_overview(df, *, text=None, width=8.0, top=None):
    """Every zero-cause pooled over the arms, with each arm's share of it.

    The per-arm notebooks each show their own causes; nothing until now showed
    the taxonomy as a whole. Rows are ordered by the total across all arms, and
    each is stacked by arm, so the figure answers two questions at once — which
    mistakes dominate the benchmark, and whether a given mistake belongs to one
    method or to all of them.

    Stage is not encoded here: the arm is the colour, and stage is the subject
    of `outcome_mix`. `top` keeps only the commonest causes when the tail is
    long.
    """
    t = _text("cause_overview", text)
    arms = ja.arm_order(df)
    colors = arm_colors(arms)
    table = ja.cause_by_arm(df)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    if top:
        table = table.head(top)
    rows = list(table.index)[::-1]
    zeros = df[df.scored & (df.reward == 0)]

    fig, ax = plt.subplots(figsize=(width, 0.30 * len(rows) + 1.9))
    left = np.zeros(len(rows))
    for arm, color in zip(arms, colors):
        values = table.loc[rows, arm].to_numpy()
        ax.barh(range(len(rows)), values, left=left, height=0.7, color=color,
                zorder=2)
        left += values
    for i, row in enumerate(rows):
        ax.annotate(f"{int(table.loc[row].sum())}", xy=(left[i], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_2)

    ax.set_xlabel(t["x"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([de(row) for row in rows], fontsize=8.5)
    ax.set_xlim(0, left.max() * 1.14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
              labels=arms, loc="lower right", ncol=1)
    tidy(ax, grid_axis="x")
    title(fig, t["title"], n=len(zeros), n_causes=zeros.fault_cause.nunique())
    fig.tight_layout()
    return fig
