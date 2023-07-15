# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
from typing import Iterable, AbstractSet

import multimethod

from koine.colors import ANSI_COLOR_RED, ANSI_COLOR_RESET
from koine.locations import Location
from koine.source import fetch_source_text
from koine.syntax import TokenID


class KoineError(Exception):
    pass


class DiagnosticError(KoineError, abc.ABC):
    def __init__(self, location: Location):
        self.__location = location

    @property
    def location(self) -> Location:
        return self.__location

    @property
    @abc.abstractmethod
    def message(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return f"[{self.location}] {ANSI_COLOR_RED}{self.message}{ANSI_COLOR_RESET}"


class ParserError(DiagnosticError):
    def __init__(self, location: Location, actual_token: TokenID, expected_tokens: Iterable[TokenID]):
        super().__init__(location)

        self.__actual_token = actual_token
        self.__expected_tokens = set(expected_tokens)

    @property
    def actual_token(self) -> TokenID:
        return self.__actual_token

    @property
    def expected_tokens(self) -> AbstractSet[TokenID]:
        return self.__expected_tokens

    @property
    def message(self) -> str:
        expected_tokens = sorted(self.expected_tokens, key=lambda t: t.description)
        actual_name = self.actual_token.quoted_description

        if expected_tokens:
            required_names = ', '.join(x.quoted_description for x in expected_tokens)
            if len(expected_tokens) > 1:
                return f"Required one of {required_names}, but got {actual_name}"
            return f"Required {required_names}, but got {actual_name}"
        return f"Unexpected {actual_name}"

    @staticmethod
    def merge(lhs: ParserError | None, rhs: ParserError | None) -> ParserError | None:
        """
        Merge two error at longest position in source text

        :param lhs:
        :param rhs:
        :return:
        """
        if not lhs:
            return rhs
        if not rhs:
            return lhs
        if lhs.location < rhs.location:
            return rhs
        if lhs.location == rhs.location:
            expected_tokens = lhs.expected_tokens | rhs.expected_tokens
            return ParserError(lhs.location, lhs.actual_token, expected_tokens)
        return lhs


class SemanticError(DiagnosticError):
    def __init__(self, location: Location, message: str):
        super().__init__(location)

        self.__message = message

    @property
    def message(self) -> str:
        return self.__message


@multimethod.multimethod
def fetch_error_string(error: KoineError) -> str:
    return str(error)


@multimethod.multimethod
def fetch_error_string(error: DiagnosticError) -> str:
    try:
        source = fetch_source_text(error.location.filename)
    except IOError:
        return str(error)
    else:
        return source.get_message(error.location, error.message)
