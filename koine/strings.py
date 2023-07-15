# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import re


def quote_identifier(name: object) -> str:
    """ Returns quoted string """
    return f'‘{name}’'


def camel_case_to_lower(name: str) -> str:
    """ Returns string converted from camel case to lower case with whitespaces """
    parts = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    return ' '.join(map(str.lower, parts))


def snake_case_to_lower(name: str) -> str:
    """ Returns string converted from lower case to lower case with whitespaces """
    return name.lower().replace('_', ' ')


def snake_case_to_camel(name: str) -> str:
    """ Returns string converted from snake case to upper camel case """
    components = name.split('_')
    return ''.join(x.title() for x in components)


