# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations


def rgb_color(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def hex_color(hex: int) -> str:
    r = (hex >> 16) & 0xFF
    g = (hex >> 8) & 0xFF
    b = hex & 0xFF
    return rgb_color(r, g, b)


ANSI_COLOR_RESET = "\x1b[0m"
ANSI_COLOR_RED = hex_color(0xff0000)
ANSI_COLOR_GREEN = hex_color(0x00ff00)
ANSI_COLOR_BLUE = hex_color(0x0000ff)
ANSI_COLOR_CYAN = hex_color(0x00ffff)
ANSI_COLOR_ORANGE = hex_color(0xff9900)
