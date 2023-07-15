# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import contextlib
from typing import Iterable, MutableMapping, Callable

import attr

from koine.exceptions import ParserError
from koine.priorities import *
from koine.scanner import fetch_syntax_tokens
from koine.syntax import *


def fetch_syntax_tree(source: SourceText) -> SyntaxTree:
    parser = Parser(source, fetch_syntax_tokens(source))
    try:
        match_module.match(parser)
    except ParserError:
        raise parser.last_error

    root = parser.pop()
    return SyntaxTree(source, root)


class Parser:
    def __init__(self, source: SourceText, tokens: Sequence[InternalToken], builder: SyntaxBuilder = None):
        self.__builder = builder or SyntaxBuilder()
        self.source = source
        self.tokens = tokens
        self.offset = 0
        self.position = 0
        self.last_error: ParserError | None = None
        self.__children = []
        self.__prefix = None

    @property
    def current(self) -> InternalToken:
        return self.tokens[self.position]

    @property
    def children(self):
        return self.__children

    def error(self, indexes: Iterable[TokenID] = None) -> ParserError:
        offset = self.offset + self.current.leading_offsets[-1]
        length = len(self.current.value)
        location = self.source.get_location(offset, length)
        error = ParserError(location, self.current.id, indexes or ())
        self.last_error = ParserError.merge(self.last_error, error)
        return error

    @contextlib.contextmanager
    def stack(self, node_class: Type[SyntaxNode]):
        """
        After execution of match block create node with children.

        :param node_class:  Node class
        """
        previous_children = self.__children
        previous_prefix = self.__prefix

        children = [previous_prefix] if previous_prefix else []

        self.__prefix = None
        self.__children = children  # new stack

        try:
            yield
        finally:
            self.__children = previous_children
            self.__prefix = previous_prefix

        self.new(node_class, children)

    @contextlib.contextmanager
    def prefix(self, prefix: InternalChild):
        previous_prefix = self.__prefix
        self.__prefix = prefix

        try:
            yield
        finally:
            self.__prefix = previous_prefix

    def new(self, node_class: Type[SyntaxNode], children: Sequence[InternalChild] = None):
        node = self.__builder.make_node(node_class, children or ())
        self.push(node)  # push

    def consume_context(self, index: TokenID):
        assert index in CONTEXTUAL_KEYWORDS
        if self.current.id != TokenID.Identifier or self.current.value != index.description:
            raise self.error((index,))

        trivia = self.__builder.make_trivia(index, self.current.value)
        token = self.__builder.make_token(trivia, self.current.leading_trivia, self.current.trailing_trivia)
        self.push(token)
        self.advance()

    def consume(self, index: TokenID):
        if self.current.id != index:
            raise self.error((index,))

        self.push(self.current)
        self.advance()

    def advance(self):
        self.offset += self.current.length
        self.position += 1

    def push(self, node: InternalChild):
        self.__children.append(node)

    def pop(self) -> InternalChild:
        return self.__children.pop()


class Parselet(abc.ABC):
    @abc.abstractmethod
    def match(self, parser: Parser, rbp: int = 0):
        raise NotImplementedError


class CombinatorParselet(Parselet):
    def __init__(self, node_class: Type[SyntaxNode] | None, combinator: Combinator):
        self.node_class = node_class
        self.combinator = combinator

    def match(self, parser: Parser, rbp: int = 0):
        if self.node_class:
            with parser.stack(self.node_class):
                self.combinator.match(parser)
        else:
            self.combinator.match(parser)


class AlternativeParselet(Parselet):
    def __init__(self):
        self.__parselets = []

    def add_prefix(self, parselet: Parselet):
        self.__parselets.append(parselet)

    def match(self, parser: Parser, rbp: int = 0):
        last_error = None

        for parselet in self.__parselets:
            try:
                parselet.match(parser)
            except ParserError as ex:
                last_error = ParserError.merge(last_error, ex)
            else:
                return

        raise last_error or parser.error()


class TableParselet(Parselet):
    def __init__(self):
        self.__prefixes: MutableMapping[TokenID, Parselet] = {}

    @property
    def prefixes(self) -> Sequence[TokenID]:
        return tuple(self.__prefixes.keys())

    def add_prefix(self, index_or_indexes: TokenID | Iterable[TokenID], parselet: Parselet):
        indexes = (index_or_indexes,) if isinstance(index_or_indexes, TokenID) else index_or_indexes
        for token_id in indexes:
            assert token_id not in self.__prefixes
            self.__prefixes[token_id] = parselet

    def match(self, parser: Parser, rbp: int = 0):
        if prefix_parselet := self.__prefixes.get(parser.current.id):
            prefix_parselet.match(parser)
            return

        raise parser.error(set(self.__prefixes.keys()))


class PrattParselet(Parselet):
    def __init__(self):
        self.__prefixes: MutableMapping[TokenID, Parselet] = {}
        self.__postfixes: MutableMapping[TokenID, Tuple[int, Parselet]] = {}

    @property
    def prefixes(self) -> Sequence[TokenID]:
        return tuple(self.__prefixes.keys())

    def add_prefix(self, index_or_indexes: TokenID | Iterable[TokenID], parselet: Parselet):
        indexes = (index_or_indexes,) if isinstance(index_or_indexes, TokenID) else index_or_indexes
        for token_id in indexes:
            assert token_id not in self.__prefixes
            self.__prefixes[token_id] = parselet

    def add_postfix(self, index_or_indexes: TokenID | Iterable[TokenID], lbp: int, parselet: Parselet):
        assert BINDING_MIN <= lbp <= BINDING_MAX
        indexes = (index_or_indexes,) if isinstance(index_or_indexes, TokenID) else index_or_indexes
        for token_id in indexes:
            assert token_id not in self.__postfixes
            self.__postfixes[token_id] = (lbp, parselet)

    def match(self, parser: Parser, rbp: int = 0):
        if prefix_parselet := self.__prefixes.get(parser.current.id):
            # prefix
            prefix_parselet.match(parser)
        else:
            # error
            raise parser.error(set(self.__prefixes.keys()))

        # postfix
        while postfix_entry := self.__postfixes.get(parser.current.id):
            lbp, postfix_parselet = postfix_entry
            if rbp >= lbp:
                break

            with parser.prefix(parser.pop()):
                postfix_parselet.match(parser)

        # mark error
        expected_indexes = {token_id for token_id, (lbp, _) in self.__postfixes.items() if rbp < lbp}
        parser.error(expected_indexes)


@attr.dataclass(frozen=True, slots=True)
class Combinator(abc.ABC):
    @abc.abstractmethod
    def match(self, parser: Parser):
        raise NotImplementedError

    @abc.abstractmethod
    def fill(self, parser: Parser):
        raise NotImplementedError


@attr.dataclass(frozen=True, slots=True)
class TokenCombinator(Combinator):
    token_id: TokenID

    def match(self, parser: Parser):
        parser.consume(self.token_id)

    def fill(self, parser: Parser):
        parser.push(None)


@attr.dataclass(frozen=True, slots=True)
class ContextTokenCombinator(TokenCombinator):
    def match(self, parser: Parser):
        parser.consume_context(self.token_id)


@attr.dataclass(frozen=True, slots=True)
class ReferenceCombinator(Combinator):
    parselet: Parselet
    rbp: int

    def match(self, parser: Parser):
        self.parselet.match(parser, self.rbp)

    def fill(self, parser: Parser):
        parser.push(None)


@attr.dataclass(frozen=True, slots=True)
class NoneCombinator(Combinator):
    def match(self, parser: Parser):
        parser.push(None)

    def fill(self, parser: Parser):
        parser.push(None)


@attr.dataclass(frozen=True, slots=True)
class SequenceCombinator(Combinator):
    combinators: Sequence[Combinator]

    def match(self, parser: Parser):
        position = parser.position
        offset = parser.offset
        try:
            for combinator in self.combinators:
                combinator.match(parser)
        except ParserError:
            parser.position = position
            parser.offset = offset
            raise

    def fill(self, parser: Parser):
        for combinator in self.combinators:
            combinator.fill(parser)


@attr.dataclass(frozen=True, slots=True)
class OptionalCombinator(Combinator):
    combinator: Combinator

    def match(self, parser: Parser):
        try:
            self.combinator.match(parser)
        except ParserError:
            self.combinator.fill(parser)

    def fill(self, parser: Parser):
        self.combinator.fill(parser)


@enum.unique
class RepeatMode(enum.IntEnum):
    ZeroOrMany = 0
    ZeroOrOne = 1
    OneOrMany = 2

    @property
    def repeat_limits(self) -> Tuple[int, int]:
        min_count = 1 if (self == RepeatMode.OneOrMany) else 0
        max_count = 1 if (self == RepeatMode.ZeroOrOne) else sys.maxsize
        return min_count, max_count


@attr.dataclass(frozen=True, slots=True)
class EmptySequenceCombinator(Combinator):
    def match(self, parser: Parser):
        self.fill(parser)

    def fill(self, parser: Parser):
        # () => []
        parser.new(SyntaxSequence)


@attr.dataclass(frozen=True, slots=True)
class RepeatCombinator(Combinator):
    combinator: Combinator
    mode: RepeatMode

    def match(self, parser: Parser):
        min_count, max_count = self.mode.repeat_limits
        count = 0

        # { combinator } => []
        with parser.stack(SyntaxSequence):
            while True:
                try:
                    self.combinator.match(parser)
                except ParserError:
                    if min_count <= count < max_count:
                        break
                    raise
                else:
                    count += 1

    def fill(self, parser: Parser):
        # () => []
        min_count, _ = self.mode.repeat_limits
        with parser.stack(SyntaxSequence):
            for _ in range(min_count):
                self.combinator.fill(parser)


@attr.dataclass(frozen=True, slots=True)
class SeparateCombinator(Combinator):
    node_class: Type[SyntaxNode]
    combinator: Combinator
    separator: Combinator
    mode: RepeatMode

    def match(self, parser: Parser):
        min_count, max_count = self.mode.repeat_limits
        count = 0

        # { combinator / separator } => []
        with parser.stack(SyntaxSequence):
            while True:
                try:
                    with parser.stack(self.node_class):
                        # combinator
                        self.combinator.match(parser)

                        # separator
                        try:
                            self.separator.match(parser)
                        except ParserError:
                            if min_count <= count < max_count:
                                self.separator.fill(parser)
                                break
                            raise
                except ParserError:
                    if min_count <= count < max_count:
                        break
                    raise
                else:
                    count += 1

    def fill(self, parser: Parser):
        # () => []
        min_count, _ = self.mode.repeat_limits
        with parser.stack(SyntaxSequence):
            for _ in range(min_count):
                with parser.stack(self.node_class):
                    self.combinator.fill(parser)
                    self.separator.fill(parser)


@attr.dataclass(frozen=True, slots=True)
class CallableCombinator(Combinator):
    callable: Callable[[Parser], None]

    def match(self, parser: Parser):
        self.callable(parser)

    def fill(self, parser: Parser):
        raise RuntimeError('Can not use callable combinator in optional context')


Declaration = Combinator | Parselet | TokenID | Sequence['Declaration']


def flat_declaration(decl: Declaration) -> Combinator:
    if isinstance(decl, Combinator):
        return decl
    if isinstance(decl, Parselet):
        return make_reference(decl)
    if isinstance(decl, TokenID):
        return make_token(decl)
    if isinstance(decl, list):
        return make_sequence(*decl)

    raise RuntimeError('Required combinator declaration, e.g. combinator, token, parselet or list')


def flat_map(fn: Callable[[Declaration], Sequence[Combinator]], collection: Sequence[Declaration]) \
        -> Iterable[Combinator]:
    """ Map a function over a collection and flatten the result by one-level """
    return itertools.chain.from_iterable(map(fn, map(flat_declaration, collection)))


def flat_combinators(combs: Sequence[Declaration], kind: Type[SequenceCombinator]) -> Iterable[Combinator]:
    """ Map a function over a collection and flatten the result by one-level """
    return flat_map(lambda c: c.combinators if isinstance(c, kind) else [c], combs)


def make_parselet(node_class: Type[SyntaxNode] | None, combinator: Declaration) -> CombinatorParselet:
    assert node_class is None or isinstance(node_class, type)
    return CombinatorParselet(node_class, flat_declaration(combinator))


def make_token(token_id: TokenID) -> TokenCombinator:
    return TokenCombinator(token_id)


def make_context_token(token_id: TokenID) -> ContextTokenCombinator:
    return ContextTokenCombinator(token_id)


def make_reference(rule: Parselet, rbp: int = BINDING_DEFAULT) -> ReferenceCombinator:
    return ReferenceCombinator(rule, rbp)


def make_none() -> NoneCombinator:
    return NoneCombinator()


def make_sequence(*combinators: Declaration) -> Combinator | SequenceCombinator:
    combinators = tuple(flat_combinators(combinators, SequenceCombinator))
    match len(combinators):
        case 0:
            raise RuntimeError(f'Can not create sequence combinator for empty sequence')
        case 1:
            return combinators[0]
        case _:
            return SequenceCombinator(combinators)


def make_empty_collection() -> EmptySequenceCombinator:
    return EmptySequenceCombinator()


def make_repeat(*combinators: Declaration, mode: RepeatMode = RepeatMode.ZeroOrMany) -> RepeatCombinator:
    combinator = make_sequence(*combinators)
    return RepeatCombinator(combinator, mode)


def make_separate(
        node_class: Type[SyntaxNode],
        *combinators: Declaration,
        separator: Declaration,
        mode: RepeatMode = RepeatMode.ZeroOrMany) -> SeparateCombinator:
    assert isinstance(node_class, type)
    combinator = make_sequence(*combinators)
    separator = flat_declaration(separator)
    return SeparateCombinator(node_class, combinator, separator, mode)


def make_optional(*combinators: Declaration) -> OptionalCombinator:
    combinator = make_sequence(*combinators)
    return OptionalCombinator(combinator)


def make_callable(callable: Callable[[Parser], None]) -> CallableCombinator:
    return CallableCombinator(callable)


# === Helpers ----------------------------------------------------------------------------------------------------------
def make_single_statement(node_class: Type[SyntaxNode], token_id: TokenID) -> CombinatorParselet:
    # single_statement := <token_id> Newline
    return make_parselet(node_class, [token_id, TokenID.Newline])


def make_simple_statement(node_class: Type[SyntaxNode], token_id: TokenID) -> CombinatorParselet:
    # simple_statement := <token_id> ':' block
    return make_parselet(node_class, [token_id, TokenID.Colon, match_block_statement])


def make_if_statement(node_class: Type[SyntaxNode], token_id: TokenID) -> CombinatorParselet:
    # if_statement := <token_id> expression ':' block_statement [ elif_statement ]
    return make_parselet(node_class, [
        token_id,
        match_expression,
        TokenID.Colon,
        match_block_statement,
        make_optional(match_elif_statement)
    ])


def make_postfix_expression(node_class: Type[SyntaxNode], token_id: TokenID, rbp: int) -> CombinatorParselet:
    return make_parselet(node_class, [
        token_id,
        make_reference(match_expression, rbp)
    ])


def make_postfix_type(node_class: Type[SyntaxNode], token_id: TokenID, rbp: int) -> CombinatorParselet:
    return make_parselet(node_class, [
        token_id,
        make_reference(match_type, rbp)
    ])


def make_type_declaration(node_class: Type[SyntaxNode], token_id: TokenID) -> CombinatorParselet:
    # type_declaration:
    #   <token_id> Identifier [ '[' generic_parameters ']' ] [ '(' { type / ',' } ')' ] ':' type_members
    return make_parselet(node_class, [
        token_id,
        TokenID.Identifier,
        make_optional(match_generic_parameters),
        make_optional(match_type_bases),
        TokenID.Colon,
        match_type_members,
    ])


def make_node_list(tuple_class: Type[SyntaxNode], separated_class: Type[SyntaxNode], node_item: Declaration):
    node_list = AlternativeParselet()

    # list := { <item> / ',' }+
    node_list.add_prefix(
        make_parselet(tuple_class, make_sequence(
            make_none(),
            make_separate(separated_class, node_item, separator=TokenID.Comma, mode=RepeatMode.OneOrMany),
            make_none(),
        ))
    )

    # list := <item> ','
    node_list.add_prefix(
        make_parselet(tuple_class, make_sequence(
            make_none(),
            make_parselet(separated_class, make_sequence(node_item, TokenID.Comma)),
            make_none(),
        )))

    # list := <item>
    node_list.add_prefix(node_item)

    return node_list


# === Qualified identifier ---------------------------------------------------------------------------------------------
match_qualified_name = PrattParselet()

# qualified_identifier := Identifier
match_qualified_name.add_prefix(TokenID.Identifier, make_parselet(SimpleIdentifierSyntax, [
    TokenID.Identifier
]))

# qualified_identifier := qualified_identifier '.' Identifier
match_qualified_name.add_postfix(TokenID.Dot, BINDING_MAX, make_parselet(QualifiedIdentifierSyntax, [
    TokenID.Dot,
    TokenID.Identifier
]))

# === Imports ----------------------------------------------------------------------------------------------------------
match_module_import = TableParselet()

# import_from := ...
match_import_from = AlternativeParselet()

# 'from' qualified_name 'import' '*' Newline
match_import_from.add_prefix(make_parselet(AllImportSyntax, [
    TokenID.From,
    match_qualified_name,
    TokenID.Import,
    TokenID.Star,
    TokenID.Newline,
]))

# from_alias := Identifier [ 'as' Identifier ]
match_from_alias = make_parselet(AliasSyntax, [
    TokenID.Identifier,
    make_optional(
        TokenID.As,
        TokenID.Identifier
    ),
])

# 'from' qualified_name 'import' from_alias { ',' from_alias } Newline
match_import_from.add_prefix(make_parselet(FromImportSyntax, [
    TokenID.From,
    match_qualified_name,
    TokenID.Import,
    # TODO: Require at least one element
    make_separate(SeparatedAliasSyntax, match_from_alias, separator=TokenID.Comma),
    TokenID.Newline,
]))

# module_alias := qualified_name [ 'as' Identifier ]
match_module_alias = make_parselet(ModuleAliasSyntax, [
    match_qualified_name,
    make_optional(
        TokenID.As,
        TokenID.Identifier
    )
])

# import_module := 'import' module_alias { ',' module_alias } Newline
match_import_module = make_parselet(ModuleImportSyntax, [
    TokenID.Import,
    # TODO: Require at least one element
    make_separate(SeparatedModuleAliasSyntax, match_module_alias, separator=TokenID.Comma),
    TokenID.Newline
])

# module_import := &'from' import_from
match_module_import.add_prefix(TokenID.From, match_import_from)

# module_import := &'import' import_module
match_module_import.add_prefix(TokenID.Import, match_import_module)

# === Forward declartions ----------------------------------------------------------------------------------------------

# expression_expansion := ...
match_expression_expansion = AlternativeParselet()

# === Effects ----------------------------------------------------------------------------------------------------------
match_effect = PrattParselet()

# identifier_effect := ...
match_identifier_effect = AlternativeParselet()

# identifier_effect := qualified_name '[' { effect_expansion / ',' } ']'
match_identifier_effect.add_prefix(make_parselet(ParameterizedEffectSyntax, [
    match_qualified_name,
    TokenID.LeftSquare,
    make_separate(SeparatedExpressionSyntax, match_expression_expansion, separator=TokenID.Comma),
    TokenID.RightSquare,
]))

# identifier_effect := qualified_name '...'
match_identifier_effect.add_prefix(make_parselet(IdentifierEffectSyntax, match_qualified_name))

# effect := identifier_effect
match_effect.add_prefix(TokenID.Identifier, match_identifier_effect)

# === Types ------------------------------------------------------------------------------------------------------------
match_type = PrattParselet()

# type_list := ...
match_type_list = make_node_list(TupleTypeSyntax, SeparatedTypeSyntax, match_type)

# type_expansion := ...
match_type_expansion = AlternativeParselet()

# type_expansion := Identifier '...'
match_type_expansion.add_prefix(make_parselet(ExpansionTypeSyntax, [
    match_type, TokenID.Ellipsis
]))

# type_expansion := type
match_type_expansion.add_prefix(match_type)

# identifier_type := ...
match_identifier_type = AlternativeParselet()

# identifier_type := qualified_name '[' { type_expansion / ',' } ']'
match_identifier_type.add_prefix(make_parselet(ParameterizedTypeSyntax, [
    match_qualified_name,
    TokenID.LeftSquare,
    make_separate(SeparatedExpressionSyntax, match_expression_expansion, separator=TokenID.Comma),
    TokenID.RightSquare,
]))

# identifier_type := qualified_name '...'
match_identifier_type.add_prefix(make_parselet(IdentifierTypeSyntax, match_qualified_name))

# type := identifier_type
match_type.add_prefix(TokenID.Identifier, match_identifier_type)

# parenthesis_type := ...
match_parenthesis_type = AlternativeParselet()

# parenthesis_type := '(' { type / ',' } ')' -> type
match_parenthesis_type.add_prefix(make_parselet(FunctionTypeSyntax, make_sequence(
    TokenID.LeftParenthesis,
    make_separate(SeparatedTypeSyntax, match_type, separator=TokenID.Comma),
    TokenID.RightParenthesis,
    TokenID.RightArrow,
    match_type,
)))

# parenthesis_type := '(' { type / ',' } ')'
match_parenthesis_type.add_prefix(make_parselet(TupleTypeSyntax, make_sequence(
    TokenID.LeftParenthesis,
    make_separate(SeparatedTypeSyntax, match_type, separator=TokenID.Comma),
    TokenID.RightParenthesis,
)))

# curly_type := ...
match_curly_type = AlternativeParselet()

# curly_type := '{' type_list ':' type_list '}'
match_curly_type.add_prefix(make_parselet(DictionaryTypeSyntax, make_sequence(
    TokenID.LeftCurly,
    match_type_list,
    TokenID.Colon,
    match_type_list,
    TokenID.RightCurly,
)))

# curly_type := '{' type_list '}'
match_curly_type.add_prefix(make_parselet(SetTypeSyntax, make_sequence(
    TokenID.LeftCurly,
    match_type_list,
    TokenID.RightCurly,
)))

# type := parenthesis_type
match_type.add_prefix(TokenID.LeftParenthesis, match_parenthesis_type)

# type := curly_type
match_type.add_prefix(TokenID.LeftCurly, match_curly_type)

# type := '[' type_list ']'
match_type.add_prefix(TokenID.LeftSquare, make_parselet(ArrayTypeSyntax, make_sequence(
    TokenID.LeftSquare,
    match_type_list,
    TokenID.RightSquare,
)))

# type := type '?'
match_type.add_postfix(TokenID.Question, BINDING_MIN, make_parselet(OptionalTypeSyntax, make_sequence(
    TokenID.Question,
)))

# type := type '*'
match_type.add_postfix(TokenID.Star, BINDING_MIN, make_parselet(PointerTypeSyntax, make_sequence(
    TokenID.Star,
)))

# type := type '**' => type '*' '*'
match_type.add_postfix(TokenID.DoubleStar, BINDING_MIN, make_parselet(DoublePointerTypeSyntax, make_sequence(
    TokenID.DoubleStar,
)))

# type := type '|' type
match_type.add_postfix(TokenID.VerticalLine,
                       BITWISE_OR_BINDING,
                       make_postfix_type(UnionTypeSyntax, TokenID.VerticalLine, BITWISE_OR_BINDING))

# === Generic parameters -----------------------------------------------------------------------------------------------
match_generic_parameter = AlternativeParselet()

# generic_parameter := 'effect' Identifier
match_generic_parameter.add_prefix(make_parselet(EffectGenericParameterSyntax, [
    make_context_token(TokenID.Effect),
    TokenID.Identifier,
]))

# generic_parameter := Identifier ':' type
match_generic_parameter.add_prefix(make_parselet(ValueGenericParameterSyntax, [
    TokenID.Identifier,
    TokenID.Colon,
    match_type
]))

# generic_parameter := Identifier ':' type
match_generic_parameter.add_prefix(make_parselet(VariadicGenericParameterSyntax, [
    TokenID.Identifier,
    TokenID.Ellipsis,
]))

# generic_parameter := Identifier
match_generic_parameter.add_prefix(make_parselet(TypeGenericParameterSyntax, TokenID.Identifier))

# generic_parameters! := '[' { generic_parameter / ',' } ']'
match_generic_parameters = make_sequence([
    TokenID.LeftSquare,
    # TODO: Require at least one element
    make_separate(SeparatedGenericParameterSyntax, match_generic_parameter, separator=TokenID.Comma),
    TokenID.RightSquare,
])

# === Function ---------------------------------------------------------------------------------------------------------
match_function_statement = AlternativeParselet()

# parameter := Identifier [ ':' type [ '...' ] ]
match_parameter = make_parselet(ParameterSyntax, [
    TokenID.Identifier,
    make_optional(
        TokenID.Colon,
        match_type,
        make_optional(TokenID.Ellipsis)
    ),
])

# function := 'def' Identifier [ '[' generic_parameters ']' ] '(' { parameter / ',' } ')' { effect } [ '-> type_list ] ':'
#               function_statement
match_function = make_parselet(FunctionSyntax, [
    TokenID.Def,
    TokenID.Identifier,
    make_optional(match_generic_parameters),
    TokenID.LeftParenthesis,
    make_separate(SeparatedParameterSyntax, match_parameter, separator=TokenID.Comma),
    TokenID.RightParenthesis,
    make_repeat(match_effect),
    make_optional(
        TokenID.RightArrow,
        match_type_list,
    ),
    TokenID.Colon,
    match_function_statement
])


# === Target -----------------------------------------------------------------------------------------------------------
def _match_target(parser: Parser):
    match_expression.match(parser, BINDING_MAX - 1)
    node = parser.pop()

    if not issubclass(node.node_class, TargetSyntax):
        raise parser.error({TokenID.Dot, TokenID.LeftSquare})

    parser.push(node)


match_target = make_callable(_match_target)

# target_list = ...
match_target_list = make_node_list(TupleTargetSyntax, SeparatedTargetSyntax, match_target)

# === Expressions ------------------------------------------------------------------------------------------------------
match_expression = PrattParselet()
match_comprehension = PrattParselet()

# expression_list = ...
match_expression_list = make_node_list(TupleExpressionSyntax, SeparatedExpressionSyntax, match_expression)

# for_comprehension := 'for' target_list 'in' source_list [ comprehension ]
match_for_comprehension = make_parselet(ForComprehensionSyntax, make_sequence(
    TokenID.For,
    match_target_list,
    TokenID.In,
    match_expression,
    make_optional(match_comprehension),
))

# comprehension := for_comprehension
match_comprehension.add_prefix(TokenID.For, match_for_comprehension)

# comprehension := 'if' condition [ comprehension ]
match_comprehension.add_prefix(TokenID.If, make_parselet(IfComprehensionSyntax, make_sequence(
    TokenID.If,
    match_expression,
    make_optional(match_comprehension),
)))

# expression_expansion := expression '...'
match_expression_expansion.add_prefix(make_parselet(ExpansionExpressionSyntax, [
    match_expression,
    TokenID.Ellipsis
]))

# expression_expansion := expression
match_expression_expansion.add_prefix(match_expression)

# expression := Integer
match_expression.add_prefix(TokenID.Integer, make_parselet(IntegerExpressionSyntax, TokenID.Integer))

# expression := Float
match_expression.add_prefix(TokenID.Float, make_parselet(FloatExpressionSyntax, TokenID.Float))

# expression := String
match_expression.add_prefix(TokenID.String, make_parselet(StringExpressionSyntax, TokenID.String))

# expression := Identifier
match_expression.add_prefix(TokenID.Identifier, make_parselet(IdentifierExpressionSyntax, TokenID.Identifier))

# parenthesis_expression :=
match_parenthesis_expression = AlternativeParselet()

# parenthesis_expression := '(' expression for_comprehension ')'
match_parenthesis_expression.add_prefix(make_parselet(TupleComprehensionSyntax, make_sequence(
    TokenID.LeftParenthesis,
    match_expression,
    match_for_comprehension,
    TokenID.RightParenthesis,
)))

# parenthesis_expression := '(' ')'
match_parenthesis_expression.add_prefix(make_parselet(TupleExpressionSyntax, make_sequence(
    TokenID.LeftParenthesis,
    make_empty_collection(),
    TokenID.RightParenthesis,
)))

# parenthesis_expression := '(' { expression / ',' }+ ')'
match_parenthesis_expression.add_prefix(make_parselet(TupleExpressionSyntax, make_sequence(
    TokenID.LeftParenthesis,
    make_separate(SeparatedExpressionSyntax, match_expression, separator=TokenID.Comma, mode=RepeatMode.OneOrMany),
    TokenID.RightParenthesis,
)))

# parenthesis_expression := '(' expression ')'
match_parenthesis_expression.add_prefix(make_parselet(ParenthesisExpressionSyntax, make_sequence(
    TokenID.LeftParenthesis,
    match_expression,
    TokenID.RightParenthesis,
)))

# square_expression := '[' ...
match_square_expression = AlternativeParselet()

# square_expression := '[' expression for_comprehension ']'
match_square_expression.add_prefix(make_parselet(ListComprehensionSyntax, make_sequence(
    TokenID.LeftSquare,
    match_expression,
    match_for_comprehension,
    TokenID.RightSquare,
)))

# square_expression := '[' { expression / ',' }+ ']'
match_square_expression.add_prefix(make_parselet(ListExpressionSyntax, make_sequence(
    TokenID.LeftSquare,
    make_separate(SeparatedExpressionSyntax, match_expression, separator=TokenID.Comma),
    TokenID.RightSquare,
)))

# curly_expression := '{' ...
match_curly_expression = AlternativeParselet()

# curly_expression := '{' expression ':' expression for_comprehension '}'
match_curly_expression.add_prefix(make_parselet(DictionaryComprehensionSyntax, make_sequence(
    TokenID.LeftCurly,
    match_expression,
    TokenID.Colon,
    match_expression,
    match_for_comprehension,
    TokenID.RightCurly,
)))

# curly_expression := '{' { (expression ':' expression) / ',' }+ '}'
match_curly_expression.add_prefix(make_parselet(DictionaryExpressionSyntax, make_sequence(
    TokenID.LeftCurly,
    make_separate(DictionaryElementSyntax, make_sequence(
        match_expression,
        TokenID.Colon,
        match_expression,
    ), separator=TokenID.Comma),
    TokenID.RightCurly,
)))

# curly_expression := '{' expression for_comprehension '}'
match_curly_expression.add_prefix(make_parselet(SetComprehensionSyntax, make_sequence(
    TokenID.LeftParenthesis,
    match_expression,
    match_for_comprehension,
    TokenID.RightParenthesis,
)))

# curly_expression := '{' { expression / ',' }+ '}'
match_curly_expression.add_prefix(make_parselet(SetExpressionSyntax, make_sequence(
    TokenID.LeftCurly,
    make_separate(SeparatedExpressionSyntax, match_expression, separator=TokenID.Comma),
    TokenID.RightCurly,
)))
# expression := parenthesis_expression
match_expression.add_prefix(TokenID.LeftParenthesis, match_parenthesis_expression)

# expression := square_expression
match_expression.add_prefix(TokenID.LeftSquare, match_square_expression)

# expression := curly_expression
match_expression.add_prefix(TokenID.LeftCurly, match_curly_expression)

# arguments := { expression / ',' }
match_arguments = make_separate(SeparatedExpressionSyntax, match_expression_expansion, separator=TokenID.Comma)

# expression := expression '(' arguments ')'
match_expression.add_postfix(TokenID.LeftParenthesis, BINDING_MAX, make_parselet(CallExpressionSyntax, [
    TokenID.LeftParenthesis,
    match_arguments,
    TokenID.RightParenthesis,
]))

# expression := expression '[' arguments ']'
match_expression.add_postfix(TokenID.LeftSquare, BINDING_MAX, make_parselet(SubscriptExpressionSyntax, [
    TokenID.LeftSquare,
    match_arguments,
    TokenID.RightSquare,
]))

# expression := expression '.' Identifier
match_expression.add_postfix(TokenID.Dot, BINDING_MAX, make_parselet(AttributeExpressionSyntax, [
    TokenID.Dot,
    TokenID.Identifier
]))

# expression := '+' expression
match_expression.add_prefix(TokenID.Plus, make_postfix_expression(PosExpressionSyntax, TokenID.Plus, POS_BINDING))

# expression := '-' expression
match_expression.add_prefix(TokenID.Minus, make_postfix_expression(NegExpressionSyntax, TokenID.Minus, POS_BINDING))

# expression := '~' expression
match_expression.add_prefix(TokenID.Tilde, make_postfix_expression(InvertExpressionSyntax, TokenID.Tilde, POS_BINDING))

# expression := 'not' expression
match_expression.add_prefix(TokenID.Not, make_postfix_expression(NotExpressionSyntax, TokenID.Not, BOOLEAN_NOT_BINDING))

# expression := expression '**' expression
match_expression.add_postfix(TokenID.DoubleStar, POW_LEFT_BINDING,
                             make_postfix_expression(PowExpressionSyntax, TokenID.DoubleStar, POW_RIGHT_BINDING))

# expression := expression '+' expression
match_expression.add_postfix(TokenID.Plus, ADD_BINDING,
                             make_postfix_expression(AddExpressionSyntax, TokenID.Plus, ADD_BINDING))

# expression := expression '-' expression
match_expression.add_postfix(TokenID.Minus, ADD_BINDING,
                             make_postfix_expression(SubExpressionSyntax, TokenID.Minus, ADD_BINDING))

# expression := expression '*' expression
match_expression.add_postfix(TokenID.Star, MUL_BINDING,
                             make_postfix_expression(MulExpressionSyntax, TokenID.Star, MUL_BINDING))

# expression := expression '/' expression
match_expression.add_postfix(TokenID.Slash, MUL_BINDING,
                             make_postfix_expression(DivExpressionSyntax, TokenID.Slash, MUL_BINDING))

# expression := expression '//' expression
match_expression.add_postfix(TokenID.DoubleSlash, MUL_BINDING,
                             make_postfix_expression(FloorDivExpressionSyntax, TokenID.DoubleSlash, MUL_BINDING))

# expression := expression '%' expression
match_expression.add_postfix(TokenID.Percent, MUL_BINDING,
                             make_postfix_expression(ModExpressionSyntax, TokenID.Percent, MUL_BINDING))

# expression := expression '>>' expression
match_expression.add_postfix(TokenID.RightShift, SHIFT_BINDING,
                             make_postfix_expression(RightShiftExpressionSyntax, TokenID.RightShift, SHIFT_BINDING))

# expression := expression '<<' expression
match_expression.add_postfix(TokenID.LeftShift, SHIFT_BINDING,
                             make_postfix_expression(LeftShiftExpressionSyntax, TokenID.LeftShift, SHIFT_BINDING))

# expression := expression '&' expression
match_expression.add_postfix(TokenID.Ampersand, BITWISE_AND_BINDING,
                             make_postfix_expression(BitwiseAndExpressionSyntax, TokenID.Ampersand,
                                                     BITWISE_AND_BINDING))

# expression := expression '^' expression
match_expression.add_postfix(TokenID.Circumflex, BITWISE_XOR_BINDING,
                             make_postfix_expression(BitwiseXorExpressionSyntax, TokenID.Circumflex,
                                                     BITWISE_XOR_BINDING))

# expression := expression '|' expression
match_expression.add_postfix(
    TokenID.VerticalLine, BITWISE_OR_BINDING,
    make_postfix_expression(BitwiseOrExpressionSyntax, TokenID.VerticalLine, BITWISE_OR_BINDING))

# expression := expression '==' expression
match_expression.add_postfix(TokenID.DoubleEqual, COMPARE_BINDING,
                             make_postfix_expression(EqualExpressionSyntax, TokenID.DoubleEqual, COMPARE_BINDING))

# expression := expression '>' expression
match_expression.add_postfix(TokenID.Great, COMPARE_BINDING,
                             make_postfix_expression(GreatExpressionSyntax, TokenID.Great, COMPARE_BINDING))

# expression := expression '>=' expression
match_expression.add_postfix(TokenID.GreatEqual, COMPARE_BINDING,
                             make_postfix_expression(GreatEqualExpressionSyntax, TokenID.GreatEqual, COMPARE_BINDING))

# expression := expression '<' expression
match_expression.add_postfix(TokenID.Less, COMPARE_BINDING,
                             make_postfix_expression(LessExpressionSyntax, TokenID.Less, COMPARE_BINDING))

# expression := expression '<=' expression
match_expression.add_postfix(TokenID.LessEqual, COMPARE_BINDING,
                             make_postfix_expression(LessEqualExpressionSyntax, TokenID.LessEqual, COMPARE_BINDING))

# expression := expression '!=' expression
match_expression.add_postfix(TokenID.NotEqual, COMPARE_BINDING,
                             make_postfix_expression(NotEqualExpressionSyntax, TokenID.NotEqual, COMPARE_BINDING))

# expression := expression 'or' expression
match_expression.add_postfix(TokenID.Or, BOOLEAN_OR_BINDING,
                             make_postfix_expression(LogicalOrExpressionSyntax, TokenID.Or, BOOLEAN_OR_BINDING))

# expression := expression 'and' expression
match_expression.add_postfix(TokenID.And, BOOLEAN_AND_BINDING,
                             make_postfix_expression(LogicalAndExpressionSyntax, TokenID.And, BOOLEAN_AND_BINDING))

# === Expression statements --------------------------------------------------------------------------------------------
match_expression_statement = AlternativeParselet()

# expression_statement := Identifier ':' type [ '=' expression_list ] Newline
match_expression_statement.add_prefix(make_parselet(VariableStatementSyntax, [
    TokenID.Identifier,
    TokenID.Colon,
    match_type,
    make_optional(
        TokenID.Equal,
        match_expression_list,
    ),
    TokenID.Newline,
]))

# expression_statement := target '=' expression_list Newline
match_expression_statement.add_prefix(make_parselet(AssignmentStatementSyntax, [
    match_target,
    TokenID.Equal,
    match_expression_list,
    TokenID.Newline
]))

# expression_statement := expression_list Newline
match_expression_statement.add_prefix(make_parselet(ExpressionStatementSyntax, [
    match_expression_list,
    TokenID.Newline,
]))

# === Statements -------------------------------------------------------------------------------------------------------
match_statement = TableParselet()

# block_statement := Newline Indent { statement } Dedent
match_block_statement = make_parselet(BlockStatementSyntax, [
    TokenID.Newline,
    TokenID.Indent,
    # TODO: Require at least one element
    make_repeat(match_statement),
    TokenID.Dedent,
])

# else_statement := 'else' ':' block
match_else_statement = make_simple_statement(ElseStatementSyntax, TokenID.Else)

# finally_statement := 'finally' ':' block
match_finally_statement = make_simple_statement(FinallyStatementSyntax, TokenID.Finally)

# function_statement := block_statement
match_function_statement.add_prefix(match_block_statement)

# function_statement := '...' Newline
match_function_statement.add_prefix(make_parselet(EllipsisStatementSyntax, [
    TokenID.Ellipsis,
    TokenID.Newline,
]))

# statement := expression_statement
match_statement.add_prefix(match_expression.prefixes, match_expression_statement)

# statement := 'pass' Newline
match_statement.add_prefix(TokenID.Pass, make_single_statement(PassStatementSyntax, TokenID.Pass))

# statement := 'return' [ expression_list ] Newline
match_statement.add_prefix(TokenID.Return, make_parselet(ReturnStatementSyntax, [
    TokenID.Return,
    make_optional(match_expression_list),
    TokenID.Newline,
]))

# statement := 'raise' [ expression ] Newline
match_statement.add_prefix(TokenID.Raise, make_parselet(RaiseStatementSyntax, [
    TokenID.Raise,
    make_optional(match_expression),
    TokenID.Newline
]))

# statement := 'continue' Newline
match_statement.add_prefix(TokenID.Continue, make_single_statement(ContinueStatementSyntax, TokenID.Continue))

# statement := 'break' Newline
match_statement.add_prefix(TokenID.Break, make_single_statement(BreakStatementSyntax, TokenID.Break))

# statement := 'while' expression ':' block_statement
match_statement.add_prefix(TokenID.While, make_parselet(WhileStatementSyntax, [
    TokenID.While,
    match_expression,
    TokenID.Colon,
    match_block_statement
]))

# elif_statement := ...
match_elif_statement = AlternativeParselet()

# elif_statement := 'elif' expression ':' block_statement [ elif_statement ]
match_elif_statement.add_prefix(make_if_statement(IfStatementSyntax, TokenID.Elif))

# elif_statement := else_statement
match_elif_statement.add_prefix(match_else_statement)

# statement := 'if' expression ':' block_statement [ elif_statement ]
match_statement.add_prefix(TokenID.If, make_if_statement(IfStatementSyntax, TokenID.If))

# try_statement := ...
match_try_statement = AlternativeParselet()

# try_statement := 'try' ':' block_statement finally_statement
match_try_statement.add_prefix(make_parselet(TryFinallyStatementSyntax, [
    TokenID.Try,
    TokenID.Colon,
    match_block_statement,
    match_finally_statement
]))

# except_statement := 'except' ':' block_statement [ else_statement ] [ finally_statement ]
match_except_statement = make_parselet(ExceptStatementSyntax, [
    TokenID.Except,
    make_optional(
        match_type,
        make_optional(
            TokenID.As,
            TokenID.Identifier,
        )
    ),
    TokenID.Colon,
    match_block_statement
])

# try_statement := 'try' ':' block_statement [ else_statement ] [ finally_statement ]
match_try_statement.add_prefix(make_parselet(TryExceptStatementSyntax, [
    TokenID.Try,
    TokenID.Colon,
    match_block_statement,
    # TODO: Require at least one element
    make_repeat(match_except_statement),
    make_optional(match_else_statement),
    make_optional(match_finally_statement),
]))

# statement := &'try' try_statement
match_statement.add_prefix(TokenID.Try, match_try_statement)

# === Field members ----------------------------------------------------------------------------------------------------
match_field = AlternativeParselet()

# field := Identifier ':' type_list [ '=' expression_list ] Newline
match_field.add_prefix(make_parselet(FieldSyntax, [
    TokenID.Identifier,
    TokenID.Colon,
    match_type_list,
    make_optional(
        TokenID.Equal,
        match_expression_list,
    ),
    TokenID.Newline,
]))

# enum_value := ...
match_enum_value = AlternativeParselet()

# enum_value := '...'
match_enum_value.add_prefix(make_parselet(EllipsisExpressionSyntax, [
    TokenID.Ellipsis
]))

# enum_value := expression
match_enum_value.add_prefix(match_expression)

# field := Identifier '=' enum_value Newline
match_field.add_prefix(make_parselet(EnumerationConstantMemberSyntax, [
    TokenID.Identifier,
    TokenID.Equal,
    match_enum_value,
    TokenID.Newline,
]))

# === Type members -----------------------------------------------------------------------------------------------------
match_type_member = TableParselet()

# type_member := 'pass' Newline
match_type_member.add_prefix(TokenID.Pass, make_parselet(PassMemberSyntax, [
    TokenID.Pass,
    TokenID.Newline
]))

# type_member := &'def' function
match_type_member.add_prefix(TokenID.Def, match_function)

# type_member := &Identifier function
match_type_member.add_prefix(TokenID.Identifier, match_field)

# === Type declarations ------------------------------------------------------------------------------------------------
match_type_bases = make_sequence(
    TokenID.LeftParenthesis,
    make_separate(SeparatedTypeSyntax, match_type, separator=TokenID.Comma),
    TokenID.RightParenthesis,
)

# type_members := Newline Indent { type_member } Dedent
match_type_members = make_sequence([
    TokenID.Newline,
    TokenID.Indent,
    # TODO: Require at least one element
    make_repeat(match_type_member),
    TokenID.Dedent,
])

# struct := 'struct' Identifier [ '[' generic_parameters ']' ] [ '(' { type / ',' } ')' ] ':' type_members
match_struct = make_type_declaration(StructSyntax, TokenID.Struct)

# class := 'class' Identifier [ '[' generic_parameters ']' ] [ '(' { type / ',' } ')' ] ':' type_members
match_class = make_type_declaration(ClassSyntax, TokenID.Class)

# interface := 'interface' Identifier [ '[' generic_parameters ']' ] [ '(' { type / ',' } ')' ] ':' type_members
match_interface = make_type_declaration(InterfaceSyntax, TokenID.Interface)

# enum := 'enum' Identifier [ '[' generic_parameters ']' ] [ '(' { type / ',' } ')' ] ':' type_members
match_enum = make_type_declaration(EnumerationSyntax, TokenID.Enum)

# === Module members ---------------------------------------------------------------------------------------------------
match_module_member = TableParselet()

# module_member := &'def' function
match_module_member.add_prefix(TokenID.Def, match_function)

# module_member := &'struct' struct
match_module_member.add_prefix(TokenID.Struct, match_struct)

# module_member := &'class' class
match_module_member.add_prefix(TokenID.Class, match_class)

# module_member := &'interface' interface
match_module_member.add_prefix(TokenID.Interface, match_interface)

# module_member := &'enum' enum
match_module_member.add_prefix(TokenID.Enum, match_enum)

# === Module -----------------------------------------------------------------------------------------------------------
match_module: Parselet

# module := { module_import } { module_member } EndOfFile
match_module = make_parselet(ModuleSyntax, [
    make_repeat(match_module_import),
    make_repeat(match_module_member),
    TokenID.EndOfFile,
])
