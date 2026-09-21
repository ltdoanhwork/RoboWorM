#!/usr/bin/env bash
# Build a PDF preview of one appendix section (or the whole paper) without a
# system TeX install.
#
# This box has no TeX, so we use `tectonic`: a single static binary that pulls
# only the packages a document actually needs into a local cache. It is
# downloaded on first use and reused afterwards.
#
# Usage:
#   bash assets/scripts/build_paper_preview.sh appendixC   # just appendix C
#   bash assets/scripts/build_paper_preview.sh full        # the whole paper
#   bash assets/scripts/build_paper_preview.sh appendixC --png   # + page PNGs
#
# Output: assets/paper/_build/<target>.pdf (git-ignored; regenerable).
set -euo pipefail

TARGET="${1:-appendixC}"
WANT_PNG=0
for a in "$@"; do [ "$a" = "--png" ] && WANT_PNG=1; done

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PAPER="$REPO/assets/paper"
BUILD="$PAPER/_build"
TECTONIC="${TECTONIC:-/tmp/tectonic}"
TECTONIC_VER="0.15.0"
mkdir -p "$BUILD"

if [ ! -x "$TECTONIC" ]; then
  echo "==> fetching tectonic $TECTONIC_VER (no system TeX on this box)"
  url="https://github.com/tectonic-typesetting/tectonic/releases/download"
  url="$url/tectonic%40${TECTONIC_VER}/tectonic-${TECTONIC_VER}-x86_64-unknown-linux-musl.tar.gz"
  tmp="$(mktemp -d)"
  curl -sL -o "$tmp/t.tar.gz" "$url"
  tar xzf "$tmp/t.tar.gz" -C "$(dirname "$TECTONIC")"
  rm -rf "$tmp"
fi

cd "$PAPER"

if [ "$TARGET" = "full" ]; then
  SRC="iclr2027_conference.tex"
  OUTNAME="paper_full"
else
  # Section previews reuse the real preamble so that spacing, fonts and macro
  # definitions match the paper exactly; only the body is swapped for a single
  # \input. Anything else risks a preview that looks right and a paper that
  # does not.
  case "$TARGET" in
    appendixC) SECTION="appendix/3_implemetation_details"; SECNUM=2 ;;
    appendixA) SECTION="appendix/1_dataset";               SECNUM=0 ;;
    appendixB) SECTION="appendix/2_metrics";               SECNUM=1 ;;
    appendixD) SECTION="appendix/4_human_eval";            SECNUM=3 ;;
    appendixE) SECTION="appendix/5_qual";                  SECNUM=4 ;;
    *) echo "unknown target '$TARGET' (use appendixA..E or full)" >&2; exit 2 ;;
  esac
  OUTNAME="_preview_${TARGET}"
  # Preamble = everything before \begin{document}, plus the two macros that
  # live further down the main file and are used inside the appendix.
  python3 - "$SECTION" "$SECNUM" "$OUTNAME" <<'PY'
import re, sys
section, secnum, outname = sys.argv[1], int(sys.argv[2]), sys.argv[3]
main = open("iclr2027_conference.tex").read()
pre = main.split(r"\begin{document}")[0]
# Macros the appendix body uses may be defined before OR after
# \begin{document} in the main file. Carry over only the ones the preamble does
# not already have, otherwise LaTeX errors with "already defined".
macros = "\n".join(
    m.group(0) for m in re.finditer(r"\\newcommand\{\\(ours|np)\}.*", main)
    if ("\\newcommand{\\%s}" % m.group(1)) not in pre)
body = r"""
%s
%% times.sty is an OT1-era package: without this the Times fonts fail to load,
%% tectonic silently falls back to Latin Modern AND drops the bold series, so
%% every \textbf in the section renders as regular weight. The preview would
%% then misrepresent the paper.
\usepackage[OT1]{fontenc}
\begin{document}
\appendix
\renewcommand\thesection{\Alph{section}}
\numberwithin{figure}{section}
\numberwithin{table}{section}
\renewcommand{\thefigure}{\thesection\arabic{figure}}
\renewcommand{\thetable}{\thesection\arabic{table}}
\setcounter{section}{%d}
\onecolumn
\input{%s}
\end{document}
""" % (macros, secnum, section)
open(outname + ".tex", "w").write(pre + body)
PY
  SRC="$OUTNAME.tex"
fi

echo "==> compiling $SRC"
# Two passes so \cref / \pageref resolve; -k keeps going on non-fatal issues.
for _ in 1 2; do
  "$TECTONIC" -k --keep-logs -o "$BUILD" "$SRC" >/dev/null 2>"$BUILD/tectonic.err" || {
    echo "FAILED -- last lines of tectonic output:"; tail -25 "$BUILD/tectonic.err"; exit 1; }
done

PDF="$BUILD/$(basename "$SRC" .tex).pdf"
FINAL="$BUILD/${TARGET}.pdf"
[ "$PDF" = "$FINAL" ] || cp "$PDF" "$FINAL"
echo "==> wrote ${FINAL#$REPO/}"

LOG="$BUILD/$(basename "$SRC" .tex).log"
if [ -f "$LOG" ]; then
  echo "--- layout report ---"
  und=$(grep -c "Undefined control sequence\|LaTeX Warning: Reference.*undefined" "$LOG" || true)
  echo "undefined refs/macros: $und"
  echo "worst overfull hboxes:"
  grep -oE "Overfull \\\\hbox \(([0-9.]+)pt too wide" "$LOG" \
    | grep -oE "[0-9.]+" | sort -rn | head -3 | sed 's/^/  /' || echo "  none"
  grep -E "Float too large" "$LOG" | head -3 | sed 's/^/  /' || true
fi

if [ "$WANT_PNG" = "1" ]; then
  echo "--- rendering page PNGs ---"
  PYTHONPATH="${PYLIBS:-/tmp/pylibs}" python3 - "$FINAL" "$BUILD/png_$TARGET" <<'PY'
import sys, os, shutil
try:
    import pymupdf
except ImportError:
    sys.exit("pymupdf not installed: pip3 install --target /tmp/pylibs pymupdf")
pdf, outdir = sys.argv[1], sys.argv[2]
shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir)
d = pymupdf.open(pdf)
for i, p in enumerate(d):
    p.get_pixmap(dpi=115).save(os.path.join(outdir, "p%02d.png" % (i + 1)))
print("  %d pages -> %s" % (d.page_count, outdir))
PY
fi
