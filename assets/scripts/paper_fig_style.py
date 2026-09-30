"""Shared matplotlib style for paper figures.

The paper is set with `times`, so figure text must be a Times-metric serif or the
figures read as a different document. No Times/Nimbus face is installed in the venvs,
but **STIXGeneral** is Times-metric-compatible and ships with matplotlib, so it is the
face used here; `mathtext.fontset='stix'` keeps inline math in the same family instead
of falling back to DejaVu, which is the usual giveaway that a figure was not styled.

Keep in-figure prose to a minimum: a short title and the unit. Anything that needs a
sentence belongs in the LaTeX caption, where it is set in the document's own font at the
document's own size and can be read at 100% zoom.
"""

SERIF = ['STIXGeneral', 'Nimbus Roman', 'Times New Roman', 'Times', 'DejaVu Serif']


def apply(matplotlib):
    """Set the paper style on a `matplotlib` module (call before pyplot is used)."""
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': SERIF,
        'mathtext.fontset': 'stix',
        'axes.labelsize': 7,
        'axes.titlesize': 7.5,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.2,
        'ytick.major.size': 2.2,
        # Type 42 keeps text selectable/searchable in the submitted PDF; Type 3 (the
        # default) is rejected by some camera-ready checkers.
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
