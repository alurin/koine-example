# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import attr


@attr.dataclass(frozen=True, repr=False)
class Position:
    line: int = 1
    """ Line position in a document (one-based). """

    column: int = 1
    """ Character offset on a line in a document (one-based). """

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"

    def __repr__(self) -> str:
        return str(self)


@attr.dataclass(frozen=True, repr=False)
class Location:
    filename: str
    """ The location's filename """

    begin: Position = Position()
    """ The location's begin position. """

    end: Position = Position()
    """ The end's begin position. """

    def __add__(self, other: Location) -> Location:
        return Location(self.filename, self.begin, other.end)

    def __str__(self) -> str:
        if self.begin == self.end:
            return f"{self.filename}:{self.begin}"
        elif self.begin.line == self.end.line:
            return f"{self.filename}:{self.begin}-{self.end.column}"
        else:
            return f"{self.filename}:{self.begin}-{self.end}"

    def __repr__(self) -> str:
        return str(self)
