# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import sys

BINDING_DEFAULT = 0
""" Minimal power for `**` """

BINDING_MIN = 1
""" Minimal power for `**` """

BINDING_MAX = sys.maxsize
""" Maximal binding power """

POW_LEFT_BINDING = 140
""" Left binding power for `**` """

POW_RIGHT_BINDING = POW_LEFT_BINDING - 1
""" Right binding power for `**` """

POS_BINDING = 130
""" Left and right binding power for unary `+`, `-`, `~` """

MUL_BINDING = 120
""" Left and right binding power for binary `*`, `@`, `/`, `//`, `%` """

ADD_BINDING = 110
""" Left and right binding power for binary `+` and `-` """

SHIFT_BINDING = 100
""" Left and right binding power for binary `<<` and `>>` """

BITWISE_AND_BINDING = 90
""" Left and right binding power for binary `bitwise and` (`&`) """

BITWISE_XOR_BINDING = 80
""" Left and right binding power for binary `bitwise xor` (`^`) """

BITWISE_OR_BINDING = 70
""" Left and right binding power for binary `bitwise or` (`|`) """

COMPARE_BINDING = 60
""" Left and right binding power for binary `in`, `not in`, `is`, `is not`, `<`, `<=`, `>`, `>=`, `!=`, `==` """

BOOLEAN_NOT_BINDING = 50
""" Left and right binding power for unary `not` """

BOOLEAN_AND_BINDING = 40
""" Left and right binding power for binary `boolean xor` (`and`) """

BOOLEAN_OR_BINDING = 30
""" Left and right binding power for binary `boolean or` (`or`) """
