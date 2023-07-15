# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import collections
import re
import sys
import tokenize as py_tokenize
from typing import Sequence, Iterator, Tuple, Pattern, Mapping

from koine.source import SourceText
from koine.syntax import InternalToken, InternalTrivia, TokenID, OPEN_BRACKETS, CLOSE_BRACKETS, TRIVIA, KEYWORDS, \
    IMPLICITS, SyntaxBuilder


def fetch_syntax_tokens(source: SourceText) -> Sequence[InternalToken]:
    scanner = Scanner(source)
    return tuple(scanner.tokenize())


class Scanner:
    def __init__(self, source: SourceText, builder: SyntaxBuilder = None):
        self.__source = source
        self.__indentations = collections.deque([0])
        self.__is_new = True  # new line
        self.__level = 0  # disable indentation
        self.__builder = builder or SyntaxBuilder()
        self.__current = self.match_trivia(0)
        self.__position = self.__current.length

    @property
    def current_id(self) -> TokenID:
        """ Current token's identifier """
        return self.__current.id

    def tokenize(self) -> Iterator[InternalToken]:
        while True:
            leading_trivia = self.consume_trivia(False)
            trivia = self.advance_trivia()

            # new line
            if trivia.id == TokenID.Newline:
                yield self.make_token(trivia, leading_trivia)
                self.__is_new = True
                continue

            if trivia.id == TokenID.EndOfFile:
                if not self.__is_new:
                    yield self.make_token(self.make_trivia(TokenID.Newline))

                is_consumed = yield from self.emit_indentation(leading_trivia)
                if is_consumed:
                    leading_trivia = ()
                yield self.make_token(trivia, leading_trivia)
                return

            if self.__is_new:
                is_consumed = yield from self.emit_indentation(leading_trivia)
                if is_consumed:
                    leading_trivia = ()

            self.__is_new = False
            if trivia.id in OPEN_BRACKETS:
                self.__level += 1
            elif trivia.id in CLOSE_BRACKETS:
                self.__level -= 1

            trailing_trivia = self.consume_trivia(True)
            yield self.make_token(trivia, leading_trivia, trailing_trivia)

    def consume_trivia(self, is_ending: bool) -> Sequence[InternalTrivia]:
        results = []
        while True:
            # is token?
            if self.current_id not in TRIVIA:
                break  # fast exit

            # new line is token?
            if not self.__is_new and self.current_id == TokenID.Newline:
                break  # fast exit

            # advance to next trivia
            last = self.advance_trivia()
            results.append(last)

            # new line is trailing trivia?
            if is_ending and last.id == TokenID.Newline:
                break

        return tuple(results)

    def emit_indentation(self, leading_trivia: Sequence[InternalTrivia]) -> Iterator[InternalToken]:
        if leading_trivia and leading_trivia[-1].id == TokenID.Whitespace:
            trivia = leading_trivia[-1]
            leading_trivia = leading_trivia[:-1]
            indent = len(trivia.value)
        else:
            trivia = None
            indent = 0

        if self.__indentations[-1] < indent:
            yield self.make_token(self.make_trivia(TokenID.Indent, trivia.value), leading_trivia)
            self.__indentations.append(indent)
            return True

        while self.__indentations[-1] > indent:
            yield self.make_token(self.make_trivia(TokenID.Dedent))
            self.__indentations.pop()

        return False

    def advance_trivia(self) -> InternalTrivia:
        trivia = self.__current
        self.__current = self.match_trivia(self.__position)
        self.__position += self.__current.length
        return trivia

    def match_trivia(self, position: int) -> InternalTrivia:
        """
        Match next trivia

        :param position:    Current position in source text
        :return:            Tuple of new position and internal trivia
        """
        return self.match_pattern(position)

    def match_pattern(self, position: int) -> InternalTrivia:
        """
        Match trivia in source text

        :param position:    Current position in source text
        :return:          Internal trivia
        """
        if match := TOKEN_REGEXP.match(self.__source.content, position):
            value = match.group(match.lastgroup)
            token_id = TOKEN_MAPPING[match.lastgroup]
            if token_id == TokenID.Identifier:
                token_id = KEYWORDS_MAPPING.get(value, token_id)
            return self.make_trivia(token_id, value)

        if position < len(self.__source.content):
            return self.make_trivia(TokenID.Error, self.__source.content[position])
        return self.make_trivia(TokenID.EndOfFile)

    def make_token(self,
                   trivia: InternalTrivia,
                   leading_trivia: Sequence[InternalTrivia] = (),
                   trailing_trivia: Sequence[InternalTrivia] = ()) -> InternalToken:
        return self.__builder.make_token(trivia, leading_trivia, trailing_trivia)

    def make_trivia(self, token_id: TokenID, value: str = None) -> InternalTrivia:
        return self.__builder.make_trivia(token_id, sys.intern(value or ""))


def _compile_regexp(*patterns: Tuple[TokenID, str]) -> Tuple[Pattern, Mapping[str, TokenID]]:
    """ Compile trivia regexp """
    regex_parts = []
    groups = {}

    for idx, (token_id, pattern) in enumerate(patterns):
        group_name = f'GROUP{idx}'
        regex_parts.append(f'(?P<{group_name}>{pattern})')
        groups[group_name] = token_id

    return re.compile('|'.join(regex_parts)), groups


KEYWORDS_MAPPING = {key_id.description: key_id for key_id in KEYWORDS}
TOKEN_REGEXP, TOKEN_MAPPING = _compile_regexp(
    # implicit patterns
    *((key_id, re.escape(key_id.description))
      for key_id in sorted(IMPLICITS, key=lambda key_id: -len(key_id.description))),
    # string patterns
    *((TokenID.String, rf'{start_str}{end_str}')
      for start_str, end_str in sorted(py_tokenize.endpats.items(), key=lambda x: -len(x[0]))),
    # other patterns
    (TokenID.Integer, py_tokenize.Intnumber),
    (TokenID.Identifier, py_tokenize.Name),
    (TokenID.Newline, r'[\n\r]+'),
    (TokenID.Whitespace, r'[ \f\t]+'),
    (TokenID.Comment, r'#[^\r\n]*'),
    (TokenID.Error, '.'),
)
