# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from typing import Type, TypeVar, Tuple, cast

T = TypeVar('T')


def sentinel(name: str) -> Tuple[Type[T], T]:
    class SentinelType(type):
        def __str__(self) -> str:
            return name

        def __repr__(self):
            return name

        def __eq__(self, other: object) -> bool:
            return self is other

        def __hash__(self) -> int:
            return hash(type(self))

        def __bool__(self) -> bool:
            return False

    ty: Type[T] = cast(Type[T], SentinelType(name, (), {}))
    ty.__name__ = name
    return ty, ty()
