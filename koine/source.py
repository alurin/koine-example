# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import bisect
import io
import itertools
import operator
import os
import os.path
from typing import Sequence, Tuple, Iterator, TextIO, Optional

from koine.locations import Location, Position
from koine.colors import ANSI_COLOR_RED, ANSI_COLOR_RESET, ANSI_COLOR_CYAN, ANSI_COLOR_BLUE, ANSI_COLOR_GREEN

DiagnosticLines = Sequence[Tuple[int, str]]


class SourceText:
    """ An abstraction of source text. """

    def __init__(self, filename: str, content: str):
        """
        Construct new source text

        :param filename:    The path to a file where this source text is stored
        :param content:     The content of this source text
        """
        self.__filename = filename
        self.__content = content
        self.__length = len(content)
        self.__lines = content.splitlines(keepends=True)
        self.__starts = tuple(itertools.accumulate((len(line) for line in self.__lines), operator.iadd, initial=0))

    @property
    def filename(self) -> str:
        """ The path to a file where this source text is stored """
        return self.__filename

    @property
    def content(self) -> str:
        """ The content of this source text """
        return self.__content

    @property
    def lines(self) -> Sequence[str]:
        """ The sequence of lines of this source text """
        return self.__lines

    def find_line(self, position: int) -> int:
        idx = bisect.bisect_right(self.__starts, position)
        if idx:
            return idx - 1

        raise ValueError('Not found line in source text')

    def get_content(self, location: Location) -> str:
        begin_at = self.to_position(location.begin)
        end_at = self.to_position(location.end) + 1
        return self.__content[begin_at:end_at]

    def get_position(self, position: int) -> Position:
        line = self.find_line(position)
        column = position - self.__starts[line]
        return Position(line + 1, column + 1)

    def get_location(self, start: int, width: int) -> Location:
        finish = start + max(0, width - 1)
        return Location(self.filename, self.get_position(start), self.get_position(finish))

    def to_position(self, position: Position) -> int:
        return self.__starts[position.line - 1] + position.column - 1

    def get_lines(self, location: Location, before: int = 2, after: int = 2) -> DiagnosticLines:
        at_before = max(0, location.begin.line - before)
        at_after = min(len(self.lines), location.end.line + after)
        results = []
        for idx in range(at_before, at_after):
            results.append((idx + 1, self.lines[idx].rstrip("\n")))
        return results

    def get_message(self, location: Location, message: str) -> str:
        lines = self.get_lines(location)
        if lines:
            source = show_source_lines(lines, location)
            return f"[{location}] {ANSI_COLOR_RED}{message}{ANSI_COLOR_RESET}:\n{source}"
        return f"[{location}] {ANSI_COLOR_RED}{message}{ANSI_COLOR_RESET}"

    @staticmethod
    def from_string(content: str, filename: str = '') -> SourceText:
        return SourceText(filename, content)

    @classmethod
    def from_stream(cls, filename: str, stream: TextIO):
        return SourceText(filename, stream.read())

    @staticmethod
    def from_filename(filename: str) -> SourceText:
        with open(filename, encoding='utf-8') as stream:
            return SourceText.from_stream(filename, stream)

    @staticmethod
    def try_from_filename(filename: str) -> Optional[SourceText]:
        try:
            return SourceText.from_filename(filename)
        except IOError:
            return None

    @staticmethod
    def for_message(source: Optional[SourceText], location: Location) -> Optional[SourceText]:
        if source and source.filename == location.filename:
            return source
        return SourceText.try_from_filename(location.filename)

    def __len__(self) -> int:
        return self.__length

    def __getitem__(self, idx: int) -> str:
        return self.__content[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__content)


def show_source_lines(strings: DiagnosticLines, location: Location):
    """
    Convert selected lines to error message, e.g.:

    ```
        1 : from module import system =
          : --------------------------^
    ```
    """
    stream = io.StringIO()
    if not strings:
        return

    width = 5
    for idx, _ in strings:
        width = max(len(str(idx)), width)

    for line, string in strings:
        s_line = str(line).rjust(width)

        stream.write(ANSI_COLOR_CYAN)
        stream.write(s_line)
        stream.write(" : ")
        stream.write(ANSI_COLOR_BLUE)
        for column, char in enumerate(string):
            column += 1
            is_error = False
            if location.begin.line == line:
                is_error = column >= location.begin.column
            if location.end.line == line:
                is_error = is_error and column <= location.end.column

            if is_error:
                stream.write(ANSI_COLOR_RED)
            else:
                stream.write(ANSI_COLOR_GREEN)
            stream.write(char)

        stream.write(ANSI_COLOR_RESET)
        stream.write("\n")

        # write error line
        if location.begin.line <= line <= location.end.line:
            stream.write("·" * width)
            stream.write(" : ")

            for column, char in itertools.chain(enumerate(string), ((len(string), ''),)):
                column += 1

                is_error = False
                if location.begin.line == line:
                    is_error = column >= location.begin.column
                if location.end.line == line:
                    is_error = is_error and column <= location.end.column

                if is_error:
                    stream.write(ANSI_COLOR_RED)
                    stream.write("^")
                    stream.write(ANSI_COLOR_RESET)
                elif char is not None:
                    stream.write("·")
            stream.write("\n")

    return stream.getvalue()


def fetch_source_text(filename: str) -> SourceText:
    fullname = os.path.abspath(filename)
    with open(fullname, encoding='utf-8') as stream:
        return SourceText(filename, stream.read())
