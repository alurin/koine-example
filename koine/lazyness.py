# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from typing import TypeVar, Iterator, Mapping

from koine.sentinel import sentinel

_, NotInitialized = sentinel('NotInitialized')

K = TypeVar('K')
V = TypeVar('V')


class LazyMapping(Mapping[K, V]):
    def __init__(self, constructor, initializer=None, *, initial=None):
        self.__mapping = dict(initial or ())
        self.__constructor = constructor
        self.__initializer = initializer

    def __getitem__(self, key: K) -> V:
        if key is None:
            raise ValueError(u'Required key')

        value = self.__mapping.get(key, NotInitialized)
        if value is not NotInitialized:
            return value

        value = self.__constructor(key)
        self.__mapping[key] = value
        if self.__initializer:
            self.__initializer(key, value)
        return value

    def __len__(self) -> int:
        return len(self.__mapping)

    def __iter__(self) -> Iterator[K]:
        return iter(self.__mapping)
