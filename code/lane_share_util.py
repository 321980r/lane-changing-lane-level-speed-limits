# -*- coding: utf-8 -*-
"""Figure helpers: save_exact(fig, name, width_mm, folder=None) adjusts the figure width until the tight-bbox output is width_mm wide
and saves it with plot_style.save (folder=FIGW for a working figure); min_font(fig) returns the smallest font size of the visible text."""
import os
import numpy as np, pandas as pd
from plot_style import save, DATF, MM

PAD_IN = 0.02          # savefig.pad_inches set in plot_style.setup()


def _tight_w(fig):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.width + 2 * PAD_IN, bb.height + 2 * PAD_IN


def save_exact(fig, name, width_mm, tol_mm=0.15, it=10, folder=None):
    """secant iteration on the figure width; warns when the output width hardly responds (an element wider than the axes)."""
    target = width_mm * MM
    W, H = fig.get_size_inches()
    w_prev, W_prev = None, None
    for _ in range(it):
        w, h = _tight_w(fig)
        if abs(w - target) < tol_mm * MM:
            break
        slope = 1.0
        if w_prev is not None and abs(W - W_prev) > 1e-6:
            slope = (w - w_prev) / (W - W_prev)
            if slope < 0.3:
                print(f'WARNING {name}: output width responds weakly to figure width (slope {slope:.2f}); an element is wider than the axes')
                break
        w_prev, W_prev = w, W
        W = max(W + (target - w) / slope, 0.5 * target)
        fig.set_size_inches(W, H)
    w, h = _tight_w(fig)
    p = save(fig, name, folder)
    print(f'saved {p}  output {w / MM:.1f} x {h / MM:.1f} mm, min font {min_font(fig):.2f} pt')
    return p, w / MM, h / MM


def min_font(fig):
    s = []
    for t in fig.findobj(lambda o: hasattr(o, 'get_fontsize') and hasattr(o, 'get_text')):
        try:
            if t.get_visible() and str(t.get_text()).strip():
                s.append(t.get_fontsize())
        except Exception:
            pass
    return min(s) if s else np.nan
