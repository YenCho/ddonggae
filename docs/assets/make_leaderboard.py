#!/usr/bin/env python3
"""Render the SNU AI ROBOT CHALLENGE 2026 final standings as SVG (light + dark).

A stacked horizontal bar per team, segmented by match. Sorted by total, which is
also the finishing order.

Palette: categorical slots 1 and 2 of the reference data-viz palette, validated
for both surfaces (worst adjacent CVD dE 24.7 light / 26.8 dark).
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent

# rank, team, final 1, final 2
ROWS = [
    (1, "Team 14", 60, 90),
    (2, "Team 7", 70, 70),
    (3, "Team 8", 80, 30),
    (4, "Team 16", 0, 100),
    (5, "Team 10", 30, 20),
]
WINNER = "Team 14"
MAX_TOTAL = 200          # two matches, 100 each

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#84837d",
                  grid="#e6e5e1", s1="#2a78d6", s2="#eb6834", band="#f2f1ed"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a897f",
                  grid="#2e2e2c", s1="#3987e5", s2="#d95926", band="#232322"),
}

W = 840
PAD_L, PAD_R = 132, 56          # left gutter holds rank + team, right holds total
ROW_H, BAR_H = 46, 22
BAR0 = 126                       # top of the first bar, clear of the legend
BAND_PAD = 12                    # winner band bleed above/below its bar
FOOT = 80
PLOT_W = W - PAD_L - PAD_R
SEG_GAP = 2                      # surface gap between stacked segments


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(theme_name):
    t = THEMES[theme_name]
    last_bar = BAR0 + (len(ROWS) - 1) * ROW_H
    axis_y = last_bar + BAR_H + BAND_PAD + 6      # gridline bottom
    H = axis_y + FOOT
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">')
    a(f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>')

    # ---- title -------------------------------------------------------------
    a(f'<text x="28" y="40" fill="{t["ink"]}" font-size="21" font-weight="700">'
      f'SNU AI ROBOT CHALLENGE 2026 — final standings</text>')
    a(f'<text x="28" y="64" fill="{t["ink2"]}" font-size="13">'
      f'Five finalists. Final score is the sum of two matches, 100 points each.</text>')

    # ---- legend (2 series: always present) ----------------------------------
    lx = 28
    for label, col in (("Final 1", t["s1"]), ("Final 2", t["s2"])):
        a(f'<rect x="{lx}" y="80" width="10" height="10" rx="2.5" fill="{col}"/>')
        a(f'<text x="{lx + 16}" y="89" fill="{t["ink2"]}" font-size="12.5">{label}</text>')
        lx += 16 + 8.2 * len(label) + 22

    # ---- x gridlines --------------------------------------------------------
    for v in range(0, MAX_TOTAL + 1, 50):
        x = PAD_L + PLOT_W * v / MAX_TOTAL
        a(f'<line x1="{x:.1f}" y1="{BAR0 - BAND_PAD - 4}" x2="{x:.1f}" y2="{axis_y}" '
          f'stroke="{t["grid"]}" stroke-width="1"/>')
        a(f'<text x="{x:.1f}" y="{axis_y + 19}" fill="{t["ink3"]}" font-size="11.5" '
          f'text-anchor="middle">{v}</text>')

    # ---- rows ---------------------------------------------------------------
    for i, (rank, team, f1, f2) in enumerate(ROWS):
        by = BAR0 + i * ROW_H               # bar top
        win = team == WINNER
        if win:                              # band, not a colour change: hue means match
            a(f'<rect x="16" y="{by - BAND_PAD}" width="{W - 32}" '
              f'height="{BAR_H + 2 * BAND_PAD}" rx="8" fill="{t["band"]}"/>')

        a(f'<text x="30" y="{by + 16}" fill="{t["ink3"]}" font-size="14" '
          f'font-weight="{700 if win else 400}">{rank}</text>')
        a(f'<text x="54" y="{by + 16}" fill="{t["ink"]}" font-size="14.5" '
          f'font-weight="{700 if win else 500}">{esc(team)}</text>')

        w1 = PLOT_W * f1 / MAX_TOTAL
        w2 = PLOT_W * f2 / MAX_TOTAL
        x = PAD_L
        for val, w, col in ((f1, w1, t["s1"]), (f2, w2, t["s2"])):
            if val <= 0:
                continue
            a(f'<rect x="{x:.1f}" y="{by}" width="{max(w - SEG_GAP, 1):.1f}" height="{BAR_H}" '
              f'rx="4" fill="{col}"/>')
            if w > 30:                       # direct label only where it fits
                a(f'<text x="{x + (w - SEG_GAP) / 2:.1f}" y="{by + 15.5}" fill="#ffffff" '
                  f'font-size="12" font-weight="600" text-anchor="middle">{val}</text>')
            x += w

        total = f1 + f2
        a(f'<text x="{W - 28}" y="{by + 17}" fill="{t["ink"]}" font-size="17" '
          f'font-weight="{700 if win else 500}" text-anchor="end">{total}</text>')

    # ---- footnote -----------------------------------------------------------
    fy = axis_y + 48
    a(f'<text x="28" y="{fy}" fill="{t["ink2"]}" font-size="12.5">'
      f'Each bar is split into the two match scores that make up the total.</text>')
    a(f'<text x="28" y="{fy + 21}" fill="{t["ink3"]}" font-size="11.5">'
      f'Ties are broken in favour of the faster mission completion.</text>')

    a('</svg>')
    return "\n".join(o)


if __name__ == "__main__":
    for name in THEMES:
        p = OUT / f"final-standings-{name}.svg"
        p.write_text(svg(name), encoding="utf-8")
        print(f"{p.name}  {p.stat().st_size:,} bytes")
