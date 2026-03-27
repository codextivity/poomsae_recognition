"""Matplotlib font configuration helpers for Korean text on Windows."""

from pathlib import Path

import matplotlib
from matplotlib import font_manager


def configure_korean_font():
    """Configure matplotlib to use a Korean-capable font if one is available."""
    candidate_fonts = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]

    selected_name = None
    for font_path in candidate_fonts:
        if Path(font_path).exists():
            try:
                selected_name = font_manager.FontProperties(fname=font_path).get_name()
                break
            except Exception:
                continue

    if selected_name:
        matplotlib.rcParams['font.family'] = selected_name

    matplotlib.rcParams['axes.unicode_minus'] = False
    return selected_name
