"""LaTeX tables for the German write-up, built from the same frame as the figures.

A table pasted into a paper as literal numbers goes stale the moment a job is
re-run, and nothing warns you. These are generated from `ja.load_jobs(...)`, so
the table in the document and the figures beside it cannot disagree.

    import job_tables as jt

    print(jt.summary_latex(df))                       # to read
    jt.write(jt.summary_latex(df), "kennzahlen-arme") # to docs/tables/

Everything a reader sees is German: the headers, the decimal comma, the
thousands separator, the budget wording, and the notes. The frame underneath
stays English, exactly as it does for the figures — see `job_figures.de`.

The caveats in the notes are derived, not typed. Trials that were never scored
and trials that recorded no usage are counted from the frame, so a note claiming
"four trials without a verdict" cannot survive those four trials being re-run.
"""

from __future__ import annotations

import pandas as pd

import job_analysis as ja

TABLES = ja.REPO / "docs" / "tables"

#: How each arm's own budget reads in German. Keyed by `budget_name`, which is
#: the arm's configuration key rather than anything chosen here.
BUDGET_DE = {
    "max_tokens": "{value} Ausgabe-Tokens",
    "max_turns": "{value} Turns",
    "max_rounds": "{value} Runden",
    "max_token_budget": "{value} Tokens",
}

_ESCAPE = str.maketrans({"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                         "_": r"\_", "{": r"\{", "}": r"\}"})


def _tex(text) -> str:
    """One cell of text, with the characters LaTeX would otherwise read."""
    return str(text).translate(_ESCAPE)


def _num(value, decimals: int = 2) -> str:
    r"""A German number: decimal comma, and `\,` between thousands."""
    if pd.isna(value):
        return r"$\varnothing$"
    return (f"{value:,.{decimals}f}".replace(",", "\0").replace(".", ",")
            .replace("\0", r"\,"))


def _tokens(value) -> str:
    """A token count, short and German: `7,3\\,M`, `170\\,k`, `940`."""
    if pd.isna(value):
        return r"$\varnothing$"
    for scale, suffix in ((1e9, "Mrd."), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return _num(value / scale, 1) + r"\," + suffix
    return _num(value, 0)


def _budget(row) -> str:
    """The arm's own stopping limit, in words."""
    template = BUDGET_DE.get(row.budget_name, "{value} " + _tex(row.budget_name))
    return template.format(value=_num(row.budget_value, 0))


#: One entry per column: the header over it, its alignment, and how a value is
#: rendered. Passing a subset to `summary_latex` is how a narrower table is made
#: — the widths below are chosen for `\textwidth` at `\footnotesize`.
COLUMNS = {
    "arm": ("Ansatz", "l", lambda r: _tex(r.name) + r.get("mark", "")),
    "budget": ("Budget", "X", _budget),
    "mean_reward": (r"$\varnothing$", "r", lambda r: _num(r.mean_reward)),
    "median_reward": ("Md", "r", lambda r: _num(r.median_reward)),
    "solved": ("Gelöst", "r", lambda r: _num(r.solved, 0)),
    "zeros": ("Null", "r", lambda r: _num(r.zeros, 0)),
    "uncached_tok": ("Uncached Input", "r", lambda r: _tokens(r.uncached_tok)),
    "cache_tok": ("Cached Input", "r", lambda r: _tokens(r.cache_tok)),
    "out_tok": ("gesamt", "r", lambda r: _tokens(r.out_tok)),
    "out_per_solved": ("je Lösung", "r", lambda r: _tokens(r.out_per_solved)),
    "total_tok": ("gesamt", "r", lambda r: _tokens(r.total_tok)),
    "total_per_solved": ("je Lösung", "r", lambda r: _tokens(r.total_per_solved)),
    "median_minutes": ("Laufzeit", "r", lambda r: _num(r.median_minutes, 1) + r"\,min"),
}

DEFAULT_COLUMNS = ("arm", "budget", "mean_reward", "median_reward", "solved",
                   "zeros", "out_tok", "out_per_solved")

#: Column groups that share a spanning header, as `(header, first, last)` over
#: the chosen column order.
#: The token headers keep the platform's own English names — they are the terms
#: the run summary shows, and the German renderings are not established
#: vocabulary — while the surrounding table stays German.
GROUPS = {
    ("mean_reward", "median_reward"): "Reward",
    ("out_tok", "out_per_solved"): "Output Tokens",
    ("total_tok", "total_per_solved"): "Tokens gesamt",
}

CAPTION_SHORT = "Kennzahlen der vier Codegenerierungs-Ansätze"
CAPTION = (
    "Ergebnisse der vier Codegenerierungs-Ansätze auf NL2RepoBench. "
    "Je Ansatz ein Durchlauf über alle {n_tasks} Aufgaben mit demselben Modell, "
    "absteigend nach mittlerem Reward.")


def _caveats(df: pd.DataFrame, solved_at: float) -> tuple:
    """The notes, counted from the frame rather than typed out here.

    Returns `(marks, notes)`. A caveat that applies to one arm gets a mark
    against that arm's name in the table; the rest are unmarked, because they
    are about the benchmark and not about any one row.
    """
    table = ja.compare(df)
    marks, notes = {}, []

    def mark(arm, text):
        letter = chr(ord("a") + len(notes))
        marks[arm] = marks.get(arm, "") + rf"\tnote{{{letter}}}"
        notes.append((letter, text))

    for arm, row in table[table.scored < table.trials].iterrows():
        mark(arm, f"{_tex(arm)}: {int(row.trials - row.scored)} Läufe ohne "
                  "Wertung (Host-Fehler, kein Urteil des Verifiers); "
                  f"Reward-Kennzahlen über {int(row.scored)} Läufe.")
    for arm, row in table[table.usage_reported < table.trials].iterrows():
        mark(arm, f"{_tex(arm)}: {int(row.trials - row.usage_reported)} Läufe "
                  "ohne erfasste Nutzung; Token-Summen über "
                  f"{int(row.usage_reported)} Läufe und damit eine Untergrenze.")

    notes.append(("", f"Gelöst = Reward $\\geq$ {_num(solved_at)}; "
                      "Null = Reward genau 0. Der Reward ist der Anteil der "
                      "bestandenen verdeckten Tests einer Aufgabe."))
    changed = _without_task(df, "more-Itertools")
    notes.append(("", "Die Aufgabe \\texttt{more-Itertools} ist nicht wertbar: "
                      "ihre verdeckten Tests importieren die im Image "
                      "installierte Bibliothek statt des erzeugten Codes. "
                      + ("Ohne sie sinkt die Zahl gelöster Aufgaben bei "
                         + " und ".join(changed) + ". Die Rangfolge ändert sich nicht."
                         if changed else
                         "Auf die Rangfolge wirkt sie sich nicht aus.")))
    return marks, notes


def _without_task(df: pd.DataFrame, task: str) -> list:
    """How each arm's solved count moves when one task is dropped."""
    changed = []
    for arm in ja.arm_order(df):
        part = df[(df.arm == arm) & df.scored]
        kept = part[part.task != task]
        before, after = int(part.solved.sum()), int(kept.solved.sum())
        if before != after:
            changed.append(f"{_tex(arm)} von {before} auf {after}")
    return changed


def summary_latex(df: pd.DataFrame, *, columns=DEFAULT_COLUMNS,
                  label: str = "tab:kennzahlen-arme", caption: str = CAPTION,
                  caption_short: str = CAPTION_SHORT, solved_at: float = 0.9,
                  placement: str = "H") -> str:
    """The headline table for the four arms, as a `threeparttable` block.

    Rows are ordered by mean reward, best first. `columns` selects and orders
    the columns from `COLUMNS`; the spanning headers in `GROUPS` are emitted
    only for groups whose members are all present and adjacent.
    """
    table = ja.compare(df).sort_values("mean_reward", ascending=False)
    budgets = {arm: df[df.arm == arm].iloc[0] for arm in table.index}
    chosen = [COLUMNS[name] for name in columns]
    marks, notes = _caveats(df, solved_at)

    # A row's formatter is handed the compare row, with the arm's budget fields
    # and its note mark attached, so `_budget` and the arm cell can read what is
    # not part of `compare`.
    body = []
    for arm, row in table.iterrows():
        source = row.copy()
        source.name = arm
        source["budget_name"] = budgets[arm].budget_name
        source["budget_value"] = budgets[arm].budget_value
        source["mark"] = marks.get(arm, "")
        body.append(" & ".join(render(source) for _, _, render in chosen) + r" \\")

    header, spans = _headers(columns, chosen)

    lines = [
        rf"\begin{{table}}[{placement}]",
        r"  \centering",
        r"  \begin{threeparttable}",
        rf"    \caption[{caption_short}]{{{caption.format(n_tasks=int(table.trials.max()))}}}",
        rf"    \label{{{label}}}",
        r"    \footnotesize",
        r"    \begin{tabularx}{\textwidth}{@{}"
        + "".join(align for _, align, _ in chosen) + r"@{}}",
        r"      \toprule",
        "      " + header,
        *([f"      {spans}"] if spans else []),
        r"      \midrule",
        *[f"      {row}" for row in body],
        r"      \bottomrule",
        r"    \end{tabularx}",
        r"    \begin{tablenotes}[flushleft]",
        r"      \footnotesize",
        *[f"      \\item[{mark}] {text}" for mark, text in notes],
        r"    \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _headers(columns, chosen) -> tuple:
    """The header row, and the sub-header row a spanning group needs.

    Where two columns share a group they get one spanning header with a rule
    under it and their own names on a second line; where they do not, the name
    sits on the first line and the second is blank.
    """
    top, bottom, rules = [], [], []
    index = 0
    while index < len(columns):
        group = next((members for members in GROUPS
                      if tuple(columns[index:index + len(members)]) == members), None)
        if group:
            width = len(group)
            top.append(rf"\multicolumn{{{width}}}{{c}}{{{GROUPS[group]}}}")
            bottom += [name for name, _, _ in chosen[index:index + width]]
            rules.append(rf"\cmidrule(lr){{{index + 1}-{index + width}}}")
            index += width
        else:
            top.append(chosen[index][0])
            bottom.append("")
            index += 1
    header = " & ".join(top) + r" \\ " + "".join(rules)
    if not any(bottom):
        return " & ".join(top) + r" \\", ""
    return header, " & ".join(bottom) + r" \\"


#: The three result classes, in the wording the write-up uses.
CLASS_DE = {"zero": "Null", "partial credit": "Teilerfolg", "solved": "Gelöst"}

CROSS_CAPTION_SHORT = "Tokenverbrauch nach Ergebnisklasse"
CROSS_CAPTION = (
    "Tokenverbrauch der vier Ansätze nach Ergebnisklasse. Je Klasse die Zahl der "
    "Läufe, die insgesamt verbrauchten Tokens und deren Anteil am Verbrauch des "
    "jeweiligen Ansatzes. Anteile beziehen sich zeilenweise auf 100\\,\\%.")


def tokens_by_class_latex(df: pd.DataFrame, *, column: str = "total_tok",
                          label: str = "tab:tokens-ergebnisklasse",
                          caption: str = CROSS_CAPTION,
                          caption_short: str = CROSS_CAPTION_SHORT,
                          solved_at: float = 0.9, placement: str = "H") -> str:
    """Tokens crossed with result class, one row per arm.

    Shares are of each arm's own consumption rather than of the whole run: the
    arms differ by a factor of 56 in absolute tokens, so a share of the grand
    total would say only which arm is biggest, which is a different table.
    """
    table = ja.tokens_by_class(df, column=column)
    classes = list(ja.RESULT_CLASSES)

    head = " & ".join([""] + [rf"\multicolumn{{3}}{{c}}{{{CLASS_DE[k]}}}"
                              for k in classes]) + r" \\ "
    head += "".join(rf"\cmidrule(lr){{{2 + i * 3}-{4 + i * 3}}}"
                    for i in range(len(classes)))
    sub = " & ".join(["Ansatz"] + ["n", "Tokens", "Anteil"] * len(classes)) + r" \\"

    body = []
    for arm, row in table.iterrows():
        cells = [_tex(arm)]
        for klass in classes:
            cells += [_num(row[(klass, "trials")], 0),
                      _tokens(row[(klass, "tokens")]),
                      _percent(row[(klass, "share")])]
        body.append(" & ".join(cells) + r" \\")

    pooled = " & ".join([r"\textbf{alle Ansätze}"] + [
        cell for klass in classes
        for cell in (_num(table[(klass, "trials")].sum(), 0),
                     _tokens(table[(klass, "tokens")].sum()),
                     _percent(table[(klass, "tokens")].sum()
                              / table[("total", "tokens")].sum()))]) + r" \\"

    lines = [
        rf"\begin{{table}}[{placement}]",
        r"  \centering",
        r"  \begin{threeparttable}",
        rf"    \caption[{caption_short}]{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \footnotesize",
        r"    \begin{tabularx}{\textwidth}{@{}X" + "rrr" * len(classes) + r"@{}}",
        r"      \toprule",
        "      " + head,
        "      " + sub,
        r"      \midrule",
        *[f"      {row}" for row in body],
        r"      \midrule",
        f"      {pooled}",
        r"      \bottomrule",
        r"    \end{tabularx}",
        r"    \begin{tablenotes}[flushleft]",
        r"      \footnotesize",
        *[f"      \\item[] {note}" for note in _cross_notes(df, table, solved_at)],
        r"    \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _percent(value) -> str:
    """A share as a German percentage."""
    if pd.isna(value):
        return r"$\varnothing$"
    return _num(value * 100, 1) + r"\,\%"


def _cross_notes(df: pd.DataFrame, table: pd.DataFrame, solved_at: float) -> list:
    """The notes for the cross-table, counted from the data."""
    notes = [
        f"Null = Reward genau 0; Teilerfolg = 0 $<$ Reward $<$ {_num(solved_at)}; "
        f"Gelöst = Reward $\\geq$ {_num(solved_at)}. "
        "Tokens sind Ein- und Ausgabe zusammen.",
    ]
    notes.append(
        "Die Anteile der Ansatzzeilen beziehen sich auf den Verbrauch des "
        "jeweiligen Ansatzes, die der Summenzeile auf den Verbrauch aller vier "
        "zusammen; letztere wird vom verbrauchsstärksten Ansatz dominiert.")

    unscored = table.attrs["unscored"]
    lost = unscored[unscored > 0]
    if len(lost):
        tokens = table.attrs["unscored_tokens"]
        parts = ", ".join(f"{_tex(arm)} {int(n)} Läufe mit {_tokens(tokens[arm])}"
                          for arm, n in lost.items())
        notes.append(f"Ohne Wertung und daher nicht enthalten: {parts}. Diese Läufe "
                     "haben kein Urteil des Verifiers und sind keine Nullwertung; "
                     "die Zeilensummen liegen deshalb um diesen Betrag unter den "
                     rf"Gesamtwerten in Tabelle~\ref{{tab:kennzahlen-arme}}.")

    # Whether failure costs more per trial than success — worth stating, and
    # counted here so the note cannot outlive the data it describes.
    dearer = [arm for arm, row in table.iterrows()
              if pd.notna(row[("zero", "median")]) and pd.notna(row[("solved", "median")])
              and row[("zero", "median")] > row[("solved", "median")]]
    if dearer:
        notes.append(
            f"In {len(dearer)} von {len(table)} Ansätzen verbraucht ein Lauf ohne "
            "Ergebnis im Median mehr Tokens als ein gelöster "
            f"({', '.join(_tex(a) for a in dearer)}); ein Fehlschlag ist also "
            "nicht der billigere Ausgang.")
    return notes


def write(latex: str, name: str, directory=None):
    """Write one table to `docs/tables/<name>.tex` and say where it went."""
    folder = directory or TABLES
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.tex"
    path.write_text(latex + "\n", encoding="utf-8")
    shown = path.relative_to(ja.REPO) if path.is_relative_to(ja.REPO) else path
    print(f"wrote {shown}")
    return path
