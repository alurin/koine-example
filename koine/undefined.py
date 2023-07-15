# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from koine.sentinel import sentinel

_, Undefined = sentinel('Undefined')


def convert_and_store_undefined(current_value, new_value, converter):
    """ This helper is used for assign value only once """
    if current_value is Undefined:
        return converter(new_value)

    raise RuntimeError('Can not store already existed value')


def store_undefined(current_value, new_value):
    """ This helper is used for assign value only once """
    if current_value is Undefined:
        return new_value

    raise RuntimeError('Can not store already existed value')


def load_undefined(current_value):
    if current_value is not Undefined:
        return current_value

    raise RuntimeError('Can not load undefined value')


def unwrap_undefined(current_value, converter):
    if callable(current_value):
        return converter(current_value())
    return current_value


def wrap_undefined(current_value, new_value, converter):
    if callable(new_value):
        return store_undefined(current_value, new_value)
    return convert_and_store_undefined(current_value, new_value, converter)
