#!/usr/bin/env python3
"""gen_charts.py data.json <outdir>: static SVG bar charts (light + dark) for this README.

Palette: the dataviz reference palette, categorical slots 1-3 (validated all-pairs in both modes).
Each chart is written twice (<name>-light.svg, <name>-dark.svg) for a <picture> element.
"""
import json, sys, html

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", s=["#3987e5", "#d95926", "#199e70"]),
}
FONT = 'font-family="system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"'


def esc(s):
    return html.escape(str(s), quote=True)


def nice_max(v):
    return min(m * 10 ** e for m in (1, 2, 2.5, 3, 4, 5, 6, 8) for e in range(-2, 7) if m * 10 ** e >= v)


def ticks(vmax, n=None):
    if n is None:  # 5 or 4 intervals, whichever gives a 1/2/5 step
        lead = lambda st: int(f"{st:.6e}"[0]) if f"{st:.6e}"[1:8] == ".000000" else 0
        n = next((k for k in (5, 4) if lead(vmax / k) in (1, 2, 5)), 5)
    step = vmax / n
    return [round(step * i, 6) for i in range(n + 1)]


def fmt(v):
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def bar_end_path(x0, y, w, h, r=4):
    """Horizontal bar anchored at x0, square at the baseline, 4px rounded data end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def hbar_chart(spec, t):
    """spec: title, subtitle, unit, rows [{label, sub, value, color(slot idx or null), text}], note"""
    rows = spec["rows"]
    W, left, right = 760, 250, 150
    top = 74 if spec.get("subtitle") else 52
    rowh, barh = 44, 18
    plot_w = W - left - right
    notes = spec.get("note", "").split("\n") if spec.get("note") else []
    H = top + rowh * len(rows) + 50 + 16 * len(notes)
    vmax = nice_max(max(r["value"] for r in rows) or 1)
    sx = lambda v: left + plot_w * v / vmax
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-labelledby="t d" {FONT}>',
         f'<title id="t">{esc(spec["title"])}</title><desc id="d">{esc(spec.get("desc", spec["title"]))}</desc>',
         f'<rect width="{W}" height="{H}" rx="8" fill="{t["surface"]}"/>',
         f'<text x="20" y="30" font-size="17" font-weight="600" fill="{t["ink"]}">{esc(spec["title"])}</text>']
    if spec.get("subtitle"):
        o.append(f'<text x="20" y="52" font-size="13" fill="{t["ink2"]}">{esc(spec["subtitle"])}</text>')
    bottom = top + rowh * len(rows)
    for v in ticks(vmax):
        x = sx(v)
        o.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{bottom + 18}" font-size="12" text-anchor="middle" fill="{t["muted"]}">{fmt(v)}</text>')
    o.append(f'<text x="{left + plot_w / 2:.1f}" y="{bottom + 36}" font-size="12" text-anchor="middle" fill="{t["ink2"]}">{esc(spec["unit"])}</text>')
    for i, r in enumerate(rows):
        y = top + i * rowh + (rowh - barh) / 2
        o.append(f'<text x="{left - 12}" y="{y + 8}" font-size="13" text-anchor="end" fill="{t["ink"]}">{esc(r["label"])}</text>')
        if r.get("sub"):
            o.append(f'<text x="{left - 12}" y="{y + 24}" font-size="11" text-anchor="end" fill="{t["ink2"]}">{esc(r["sub"])}</text>')
        col = t["s"][r["color"]] if r.get("color") is not None else t["muted"]
        w = sx(r["value"]) - left
        p = bar_end_path(left, y, w, barh)
        if p:
            o.append(f'<path d="{p}" fill="{col}"><title>{esc(r["label"])}: {esc(r.get("text", fmt(r["value"])))}</title></path>')
        if r.get("lo") is not None:
            x1, x2, yc = sx(r["lo"]), sx(r["hi"]), y + barh / 2
            o.append(f'<path d="M{x1:.1f},{yc - 5:.1f} v10 M{x1:.1f},{yc:.1f} H{x2:.1f} M{x2:.1f},{yc - 5:.1f} v10" stroke="{t["ink2"]}" stroke-width="1.5" fill="none"/>')
        tx = max(left + w, sx(r["hi"]) if r.get("hi") is not None else 0) + 8
        o.append(f'<text x="{tx:.1f}" y="{y + 14}" font-size="13" fill="{t["ink"]}">{esc(r.get("text", fmt(r["value"])))}</text>')
    o.append(f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
    for ni, line in enumerate(notes):
        o.append(f'<text x="20" y="{H - 12 - 16 * (len(notes) - 1 - ni)}" font-size="11" fill="{t["muted"]}">{esc(line)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def grouped_chart(spec, t):
    """spec: title, subtitle, panels [{name, unit, values: [v per series], err: [..]}], series [names]"""
    panels, series = spec["panels"], spec["series"]
    W = 760
    pw = (W - 40) / len(panels)
    top = 110
    ph = 230
    notes = spec.get("note", "").split("\n") if spec.get("note") else []
    H = top + ph + 66 + 16 * len(notes)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-labelledby="t d" {FONT}>',
         f'<title id="t">{esc(spec["title"])}</title><desc id="d">{esc(spec.get("desc", spec["title"]))}</desc>',
         f'<rect width="{W}" height="{H}" rx="8" fill="{t["surface"]}"/>',
         f'<text x="20" y="30" font-size="17" font-weight="600" fill="{t["ink"]}">{esc(spec["title"])}</text>']
    if spec.get("subtitle"):
        o.append(f'<text x="20" y="52" font-size="13" fill="{t["ink2"]}">{esc(spec["subtitle"])}</text>')
    # legend, one row above the plots
    lx = 20
    for i, s in enumerate(series):
        o.append(f'<rect x="{lx}" y="68" width="12" height="12" rx="3" fill="{t["s"][i]}"/>')
        o.append(f'<text x="{lx + 18}" y="78.5" font-size="13" fill="{t["ink"]}">{esc(s)}</text>')
        lx += 30 + 7.4 * len(s)
    for pi, p in enumerate(panels):
        x0 = 20 + pi * pw + 50
        plot_w = pw - 80
        vmax = nice_max(max(p["values"]) * 1.12)
        sy = lambda v: top + ph - ph * v / vmax
        for v in ticks(vmax, 4):
            y = sy(v)
            o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
            o.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="{t["muted"]}">{fmt(v)}</text>')
        o.append(f'<line x1="{x0}" y1="{top + ph}" x2="{x0 + plot_w:.1f}" y2="{top + ph}" stroke="{t["axis"]}" stroke-width="1"/>')
        n = len(series)
        gap = 2
        bw = min(46, (plot_w - 40 - gap * (n - 1)) / n)
        gx = x0 + (plot_w - (bw * n + gap * (n - 1))) / 2
        for si, v in enumerate(p["values"]):
            bx = gx + si * (bw + gap)
            by = sy(v)
            h = top + ph - by
            r = min(4, bw / 2, h)
            path = (f"M{bx:.1f},{top + ph:.1f} V{by + r:.1f} Q{bx:.1f},{by:.1f} {bx + r:.1f},{by:.1f} "
                    f"H{bx + bw - r:.1f} Q{bx + bw:.1f},{by:.1f} {bx + bw:.1f},{by + r:.1f} V{top + ph:.1f} Z")
            lab = fmt(round(v, 2))
            if p.get("err"):
                lab += f" ± {fmt(round(p['err'][si], 2))}"
            o.append(f'<path d="{path}" fill="{t["s"][si]}"><title>{esc(series[si])}, {esc(p["name"])}: {lab} {esc(p["unit"])}</title></path>')
            o.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 8:.1f}" font-size="12" text-anchor="middle" fill="{t["ink"]}">{fmt(round(v, 1))}</text>')
        o.append(f'<text x="{x0 + plot_w / 2:.1f}" y="{top + ph + 22}" font-size="13" font-weight="600" text-anchor="middle" fill="{t["ink"]}">{esc(p["name"])}</text>')
        o.append(f'<text x="{x0 + plot_w / 2:.1f}" y="{top + ph + 40}" font-size="12" text-anchor="middle" fill="{t["ink2"]}">{esc(p["unit"])}</text>')
    for ni, line in enumerate(notes):
        o.append(f'<text x="20" y="{H - 12 - 16 * (len(notes) - 1 - ni)}" font-size="11" fill="{t["muted"]}">{esc(line)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    data = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    for name, spec in data.items():
        for mode, t in THEMES.items():
            svg = (grouped_chart if spec["kind"] == "grouped" else hbar_chart)(spec, t)
            open(f"{out}/{name}-{mode}.svg", "w").write(svg + "\n")


if __name__ == "__main__":
    main()
