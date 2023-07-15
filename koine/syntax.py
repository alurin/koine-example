# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import ast
import enum
import itertools
import sys
from typing import Type, Sequence, TypeVar, Tuple, Iterator, cast, Generic, overload, AbstractSet

import prettyprinter
from prettyprinter.prettyprinter import pretty_bracketable_iterable
from pyrsistent import PVector, pvector

from koine.locations import Location
from koine.source import SourceText
from koine.strings import quote_identifier


# === Syntax: tokens ---------------------------------------------------------------------------------------------------
@enum.unique
class TokenID(enum.Enum):
    make_unique_id = (lambda x: lambda: next(x))(itertools.count())

    Ampersand = make_unique_id(), '@'
    AmpersandEqual = make_unique_id(), '&='
    And = make_unique_id(), 'and'
    As = make_unique_id(), 'as'
    At = make_unique_id(), '@'
    AtEqual = make_unique_id(), '@='
    Break = make_unique_id(), 'break'
    Circumflex = make_unique_id(), '^'
    CircumflexEqual = make_unique_id(), '^='
    Class = make_unique_id(), 'class'
    Colon = make_unique_id(), ':'
    Comma = make_unique_id(), ','
    Comment = make_unique_id(), 'comment'
    Concept = make_unique_id(), 'concept'
    Continue = make_unique_id(), 'continue'
    Dedent = make_unique_id(), 'dedent'
    Def = make_unique_id(), 'def'
    Dot = make_unique_id(), '.'
    DoubleEqual = make_unique_id(), '=='
    DoubleSlash = make_unique_id(), '//'
    DoubleSlashEqual = make_unique_id(), '//='
    DoubleStar = make_unique_id(), '**'
    DoubleStarEqual = make_unique_id(), '**='
    Effect = make_unique_id(), 'effect'
    Elif = make_unique_id(), 'elif'
    Ellipsis = make_unique_id(), '...'
    Else = make_unique_id(), 'else'
    EndOfFile = make_unique_id(), 'end of file'
    Enum = make_unique_id(), 'enum'
    Equal = make_unique_id(), '='
    Error = make_unique_id(), '<error>'
    Except = make_unique_id(), 'except'
    Exclamation = make_unique_id(), '!'
    Finally = make_unique_id(), 'finally'
    Float = make_unique_id(), 'float'
    For = make_unique_id(), 'for'
    From = make_unique_id(), 'from'
    Great = make_unique_id(), '>'
    GreatEqual = make_unique_id(), '>='
    Identifier = make_unique_id(), 'identifier'
    If = make_unique_id(), 'if'
    Import = make_unique_id(), 'import'
    In = make_unique_id(), 'in'
    Indent = make_unique_id(), 'indent'
    Integer = make_unique_id(), 'integer'
    Interface = make_unique_id(), 'interface'
    Is = make_unique_id(), 'is'
    LeftCurly = make_unique_id(), '{'
    LeftParenthesis = make_unique_id(), '('
    LeftShift = make_unique_id(), '<<'
    LeftShiftEqual = make_unique_id(), '<<='
    LeftSquare = make_unique_id(), '['
    Less = make_unique_id(), '<'
    LessEqual = make_unique_id(), '<='
    Minus = make_unique_id(), '-'
    MinusEqual = make_unique_id(), '-='
    Newline = make_unique_id(), 'new line'
    Not = make_unique_id(), 'not'
    NotEqual = make_unique_id(), '!='
    Or = make_unique_id(), 'or'
    Pass = make_unique_id(), 'pass'
    Percent = make_unique_id(), '%'
    PercentEqual = make_unique_id(), '%='
    Plus = make_unique_id(), '+'
    PlusEqual = make_unique_id(), '+='
    Question = make_unique_id(), '?'
    Raise = make_unique_id(), 'raise'
    Replace = make_unique_id(), '=>'
    Return = make_unique_id(), 'return'
    RightArrow = make_unique_id(), '->'
    RightCurly = make_unique_id(), '}'
    RightParenthesis = make_unique_id(), ')'
    RightShift = make_unique_id(), '>>'
    RightShiftEqual = make_unique_id(), '>>='
    RightSquare = make_unique_id(), ']'
    Semicolon = make_unique_id(), ';'
    Slash = make_unique_id(), '/'
    SlashEqual = make_unique_id(), '/='
    Star = make_unique_id(), '*'
    StarEqual = make_unique_id(), '*='
    String = make_unique_id(), 'string'
    Struct = make_unique_id(), 'struct'
    Tilde = make_unique_id(), '~'
    TildeEqual = make_unique_id(), '~='
    Try = make_unique_id(), 'try'
    Using = make_unique_id(), 'using'
    VerticalLine = make_unique_id(), '|'
    VerticalLineEqual = make_unique_id(), '|='
    Where = make_unique_id(), 'where'
    While = make_unique_id(), 'while'
    Whitespace = make_unique_id(), 'whitespace'

    @property
    def description(self) -> str:
        return self.value[1]

    @property
    def quoted_description(self) -> str:
        return quote_identifier(self.value[1])


KEYWORDS: AbstractSet[TokenID] = {
    TokenID.And,
    TokenID.As,
    TokenID.Break,
    TokenID.Continue,
    TokenID.Def,
    TokenID.Elif,
    TokenID.Else,
    TokenID.Except,
    TokenID.Finally,
    TokenID.For,
    TokenID.From,
    TokenID.If,
    TokenID.Import,
    TokenID.In,
    TokenID.Not,
    TokenID.Or,
    TokenID.Pass,
    TokenID.Raise,
    TokenID.Return,
    TokenID.Struct,
    TokenID.Try,
    TokenID.Using,
    TokenID.While,
}
""" This set is contains keyword tokens """

CONTEXTUAL_KEYWORDS: AbstractSet[TokenID] = {
    TokenID.Concept,
    TokenID.Effect,
    TokenID.Where,
}
""" This set is contains token's identifiers that used as contextual keywords """

IMPLICITS: AbstractSet[TokenID] = {
    TokenID.Ampersand,
    TokenID.Circumflex,
    TokenID.Colon,
    TokenID.Comma,
    TokenID.Dot,
    TokenID.DoubleEqual,
    TokenID.DoubleSlash,
    TokenID.DoubleStar,
    TokenID.Ellipsis,
    TokenID.Equal,
    TokenID.Great,
    TokenID.GreatEqual,
    TokenID.LeftCurly,
    TokenID.LeftParenthesis,
    TokenID.LeftShift,
    TokenID.LeftSquare,
    TokenID.Less,
    TokenID.LessEqual,
    TokenID.Minus,
    TokenID.NotEqual,
    TokenID.Percent,
    TokenID.Plus,
    TokenID.Question,
    TokenID.RightArrow,
    TokenID.RightCurly,
    TokenID.RightParenthesis,
    TokenID.RightShift,
    TokenID.RightSquare,
    TokenID.Slash,
    TokenID.Star,
    TokenID.Tilde,
    TokenID.VerticalLine,
}
""" This set is contains implicit tokens """

TRIVIA: AbstractSet[TokenID] = {
    TokenID.Comment,
    TokenID.Whitespace,
    TokenID.Newline
}
""" This set is contains tokens' identifier that explicitly used as trivia """

OPEN_BRACKETS: AbstractSet[TokenID] = {
    TokenID.LeftCurly,
    TokenID.LeftParenthesis,
    TokenID.LeftSquare,
}
""" This set is contains tokens' identifier that used as open brackets """

CLOSE_BRACKETS: AbstractSet[TokenID] = {
    TokenID.RightCurly,
    TokenID.RightParenthesis,
    TokenID.RightSquare,
}
""" This set is contains tokens' identifier that used as close brackets """


# === Syntax: internal structs -----------------------------------------------------------------------------------------
class InternalSymbol(abc.ABC):
    @property
    @abc.abstractmethod
    def length(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_inline_location(self, source: SourceText, position: int) -> Location:
        """
        Calculate inline location occupied by syntax symbol in source code.

        Inline location is not included leading and trailing trivia
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_block_location(self, source: SourceText, position: int) -> Location:
        """
        Calculate block location occupied by syntax symbol in source code.

        Block location is included leading and trailing trivia
        """
        raise NotImplementedError

    @abc.abstractmethod
    def materialize(self, tree: SyntaxTree, parent: SyntaxSymbol | None, position: int) -> SyntaxSymbol:
        raise NotImplementedError


class InternalTrivia(InternalSymbol):
    __slots__ = ('__id', '__value')

    def __init__(self, id: TokenID, value: str = None):
        self.__id = id
        self.__value = sys.intern(value or '')

    @property
    def id(self) -> TokenID:
        return self.__id

    @property
    def value(self) -> str:
        return self.__value

    @property
    def length(self) -> int:
        return len(self.value)

    def get_inline_location(self, source: SourceText, position: int) -> Location:
        return source.get_location(position, self.length)

    def get_block_location(self, source: SourceText, position: int) -> Location:
        return self.get_inline_location(source, position)

    def materialize(self, tree: SyntaxTree, parent: SyntaxToken, position: int) -> SyntaxTrivia:
        return SyntaxTrivia(tree, parent, self, position)


class InternalToken(InternalSymbol):
    __slots__ = (
        '__token_trivia',
        '__leading_trivia',
        '__leading_offsets',
        '__trailing_trivia',
        '__trailing_offsets',
        '__length'
    )

    def __init__(self,
                 token_trivia: InternalTrivia,
                 leading_trivia: Sequence[InternalTrivia] = None,
                 trailing_trivia: Sequence[InternalTrivia] = None) -> None:
        assert token_trivia

        self.__token_trivia = token_trivia
        self.__leading_trivia = leading_trivia or ()
        self.__leading_offsets = get_trivia_offsets(self.__leading_trivia)
        self.__trailing_trivia = trailing_trivia or ()
        self.__trailing_offsets = get_trivia_offsets(self.__trailing_trivia,
                                                     self.__leading_offsets[-1] + token_trivia.length)
        self.__length = self.__trailing_offsets[-1]

    @property
    def id(self) -> TokenID:
        return self.__token_trivia.id

    @property
    def value(self) -> str:
        return self.__token_trivia.value

    @property
    def length(self) -> int:
        return self.__length

    @property
    def token_trivia(self) -> InternalTrivia:
        return self.__token_trivia

    @property
    def leading_trivia(self) -> Sequence[InternalTrivia]:
        return self.__leading_trivia

    @property
    def leading_offsets(self) -> Sequence[int]:
        return self.__leading_offsets

    @property
    def trailing_trivia(self) -> Sequence[InternalTrivia]:
        return self.__trailing_trivia

    @property
    def trailing_offsets(self) -> Sequence[int]:
        return self.__trailing_offsets

    @property
    def children(self) -> Iterator[InternalTrivia]:
        yield from self.leading_trivia
        yield self.token_trivia
        yield from self.trailing_trivia

    def get_inline_location(self, source: SourceText, position: int) -> Location:
        return source.get_location(position + self.__leading_offsets[-1], self.token_trivia.length)

    def get_block_location(self, source: SourceText, position: int) -> Location:
        return source.get_location(position, self.length)

    def materialize(self, tree: SyntaxTree, parent: SyntaxNode, position: int) -> SyntaxToken:
        return SyntaxToken(tree, parent, self, position)

    def __str__(self) -> str:
        return f'{self.id.name}: {repr(self.value)} [{self.length}...]'


InternalChildren = Sequence[InternalSymbol | None]


class InternalNode(InternalSymbol):
    def __init__(self, node_class: Type[SyntaxNode], children: InternalChildren):
        assert all(isinstance(child, InternalSymbol) for child in children if child is not None)

        self.__node_class = node_class
        self.__children = pvector(children)
        self.__offsets = tuple(itertools.accumulate((s.length if s else 0 for s in children), initial=0))

    @property
    def node_class(self) -> Type[SyntaxNode]:
        return self.__node_class

    @property
    def children(self) -> PVector[InternalSymbol | None]:
        return self.__children

    @property
    def length(self) -> int:
        return self.__offsets[-1]

    @property
    def offsets(self) -> Sequence[int]:
        return self.__offsets

    def get_inline_location(self, source: SourceText, position: int) -> Location:
        return self.get_block_location(source, position)

    def get_block_location(self, source: SourceText, position: int) -> Location:
        return source.get_location(position, self.length)

    def materialize(self, tree: SyntaxTree, parent: SyntaxNode | None, position: int) -> SyntaxNode:
        return self.__node_class(tree, parent, self, position)

    def materialize_child(self, parent: SyntaxNode, index: int) -> SyntaxSymbol | None:
        internal = self.__children[index]
        if not internal:
            return None

        offset = self.__offsets[index]
        return parent.tree.materialize(parent, internal, parent.position + offset)


def get_trivia_offsets(trivia_seq: Sequence[InternalTrivia], initial: int = 0) -> Sequence[int]:
    return tuple(itertools.accumulate((trivia.length for trivia in trivia_seq), initial=initial))


@prettyprinter.register_pretty(InternalTrivia)
def pretty_internal_trivia(value: InternalTrivia, ctx):
    return prettyprinter.pretty_call_alt(ctx, type(value), args=(value.id, value.value))


@prettyprinter.register_pretty(InternalToken)
def pretty_internal_token(value: InternalToken, ctx):
    kwargs = {'id': value.id}
    if value.value:
        kwargs['value'] = value.value
    if value.leading_trivia:
        kwargs['leading_trivia'] = value.leading_trivia
    if value.trailing_trivia:
        kwargs['trailing_trivia'] = value.trailing_trivia

    return prettyprinter.pretty_call_alt(ctx, type(value), kwargs=kwargs)


@prettyprinter.register_pretty(InternalNode)
def pretty_internal_node(value: InternalNode, ctx):
    return prettyprinter.pretty_call_alt(ctx, type(value), kwargs={
        'node_class': value.node_class,
        'children': tuple(value.children),
    })


# === Syntax: concrete structs -----------------------------------------------------------------------------------------
class SyntaxSymbol(abc.ABC):
    __slots__ = ('__tree', '__position', '__diagnostics')

    def __init__(self, tree: SyntaxTree, position: int):
        self.__tree = tree
        self.__position = position

    @property
    def tree(self) -> SyntaxTree:
        return self.__tree

    @property
    @abc.abstractmethod
    def parent(self) -> SyntaxNode | None:
        raise NotImplementedError

    @property
    def position(self) -> int:
        return self.__position

    @property
    @abc.abstractmethod
    def internal(self) -> InternalSymbol:
        raise NotImplementedError

    @property
    def location(self) -> Location:
        return self.inline_location

    @property
    def block_location(self) -> Location:
        return self.internal.get_block_location(self.__tree.source, self.position)

    @property
    def inline_location(self) -> Location:
        return self.internal.get_inline_location(self.__tree.source, self.position)


class SyntaxTrivia(SyntaxSymbol):
    def __init__(self, tree: SyntaxTree, parent: SyntaxToken, internal: InternalTrivia, position: int):
        super(SyntaxTrivia, self).__init__(tree, position)

        self.__parent = parent
        self.__internal = internal

    @property
    def parent(self) -> SyntaxToken:
        return self.__parent

    @property
    def internal(self) -> InternalTrivia:
        return self.__internal

    @property
    def id(self) -> TokenID:
        return self.__internal.id

    @property
    def value(self) -> str:
        return self.__internal.value


class SyntaxToken(SyntaxSymbol):
    __slots__ = ('__parent', '__internal')

    def __init__(self, tree: SyntaxTree, parent: SyntaxNode, internal: InternalToken, position: int):
        super(SyntaxToken, self).__init__(tree, position)
        self.__parent = parent
        self.__internal = internal

    @property
    def parent(self) -> SyntaxNode:
        return self.__parent

    @property
    def internal(self) -> InternalToken:
        return self.__internal

    @property
    def id(self) -> TokenID:
        return self.__internal.id

    @property
    def value(self) -> str:
        return self.__internal.value

    @property
    def leading_trivia(self) -> Sequence[SyntaxTrivia]:
        return tuple(
            trivia.materialize(self.tree, self, self.position + self.internal.leading_offsets[idx]) for idx, trivia in
            enumerate(self.internal.leading_trivia))

    @property
    def trailing_trivia(self) -> Sequence[SyntaxTrivia]:
        return tuple(
            trivia.materialize(self.tree, self, self.position + self.internal.trailing_offsets[idx]) for idx, trivia in
            enumerate(self.internal.trailing_trivia))


class _SyntaxNodeMeta(abc.ABCMeta):
    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: dict):
        counter = itertools.count()
        attributes = {}
        for key, value in namespace.get('__annotations__', {}).items():
            if key.startswith('__'):
                continue
            index = next(counter)
            attributes[key] = property((lambda i: lambda s: s.internal.materialize_child(s, i))(index))
        if attributes:
            namespace.update(attributes)

        attributes = tuple(itertools.chain(*(getattr(base, '__attributes__', ()) for base in bases), attributes.keys()))
        namespace['__attributes__'] = attributes

        node_class = super(_SyntaxNodeMeta, mcs).__new__(mcs, name, bases, namespace)
        prettyprinter.register_pretty(node_class)(make_node_prettier(attributes))
        return node_class


def make_node_prettier(attributes: Sequence[str]):
    def node_prettier(value: SyntaxNode, ctx):
        return prettyprinter.pretty_call_alt(ctx, type(value), kwargs={
            name: getattr(value, name) for name in attributes
        })

    return node_prettier


class SyntaxNode(SyntaxSymbol, abc.ABC, metaclass=_SyntaxNodeMeta):
    __slots__ = ('__parent', '__internal')

    def __init__(self, tree: SyntaxTree, parent: SyntaxNode | None, internal: InternalNode, position: int):
        super(SyntaxNode, self).__init__(tree, position)

        self.__parent = parent
        self.__internal = internal

    @property
    def parent(self) -> SyntaxNode | None:
        return self.__parent

    @property
    def internal(self) -> InternalNode:
        return self.__internal

    @property
    def children(self) -> Sequence[SyntaxSymbol | None]:
        return [self._materialize_child(index) for index in range(len(self.__internal.children))]

    def _materialize_child(self, index: int) -> SyntaxSymbol | SyntaxNode | None:
        """ Materialize child as syntax symbol """
        if child := self.internal.children[index]:
            return child.materialize(self.tree, self, self.position + self.internal.offsets[index])
        return child


T = TypeVar('T', bound=SyntaxSymbol)


class SyntaxSequence(SyntaxNode, Generic[T], Sequence[T]):
    @overload
    def __getitem__(self, i: int) -> T: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[T]: ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            indexes = itertools.islice(range(len(self)), i.start, i.start, i.step)
            return tuple(self[index] for index in indexes)

        return self.internal.children[i].materialize(self.tree, self.parent, self.position + self.internal.offsets[i])

    def __len__(self) -> int:
        return len(self.internal.children)


class SyntaxTree:
    def __init__(self, source: SourceText, root: InternalNode):
        self.__cache = {}
        self.__source = source
        self.__root = root

    @property
    def source(self) -> SourceText:
        return self.__source

    @property
    def root(self) -> ModuleSyntax:
        return cast(ModuleSyntax, self.materialize(None, self.__root, 0))

    def materialize(self, parent: SyntaxNode | None, internal: InternalSymbol | None, position: int) \
            -> SyntaxSymbol | None:
        if internal is None:
            return None

        key = (parent, internal, position)
        if result := self.__cache.get(key):
            return result

        self.__cache[key] = result = internal.materialize(self, parent, position)
        return result


@prettyprinter.register_pretty(SyntaxTrivia)
def pretty_syntax_token(value: SyntaxTrivia, ctx):
    return prettyprinter.pretty_call_alt(ctx, type(value), kwargs={
        'id': value.id,
        'value': value.value
    })


@prettyprinter.register_pretty(SyntaxToken)
def pretty_syntax_token(value: SyntaxToken, ctx):
    kwargs = {'id': value.id}
    if value.value:
        kwargs['value'] = value.value
    if value.leading_trivia:
        kwargs['leading_trivia'] = value.leading_trivia
    if value.trailing_trivia:
        kwargs['trailing_trivia'] = value.trailing_trivia

    return prettyprinter.pretty_call_alt(ctx, type(value), kwargs=kwargs)


@prettyprinter.register_pretty(SyntaxTree)
def pretty_token_attrs(value: SyntaxTree, ctx):
    return prettyprinter.pretty_call_alt(ctx, type(value), args=(value.root,))


@prettyprinter.register_pretty(SyntaxSequence)
def pretty_token_attrs(value: SyntaxSequence, ctx):
    return pretty_bracketable_iterable(tuple(value), ctx)


# === Syntax: builder --------------------------------------------------------------------------------------------------
class SyntaxBuilder:
    """
    Syntax builder is used for create internal tokens and nodes.
    """

    def __init__(self):
        self.__trivia = {}
        self.__tokens = {}
        self.__nodes = {}
        self.__collections = {}

    def make_trivia(self, id: TokenID, value: str = None) -> InternalTrivia:
        """ Create internal trivia """
        value = sys.intern(value or '')
        key = (id, value)

        if token := self.__trivia.get(key):
            return token

        token = InternalTrivia(id, value)
        self.__trivia[key] = token
        return token

    def make_token(self,
                   trivia: InternalTrivia,
                   leading_trivia: Sequence[InternalTrivia] = (),
                   trailing_trivia: Sequence[InternalTrivia] = ()) -> InternalToken:
        """ Create internal  token """
        key = (trivia.id, trivia.value, leading_trivia, trailing_trivia)

        if token := self.__tokens.get(key):
            return token

        token = InternalToken(trivia, leading_trivia, trailing_trivia)
        self.__tokens[key] = token
        return token

    def make_node(self, node_class: Type[SyntaxNode], children: Sequence[InternalChild]) -> InternalNode:
        """ Create internal node """
        children = tuple(children)
        key = (node_class, children)

        if node := self.__nodes.get(key):
            return node

        node = InternalNode(node_class, children)
        self.__nodes[key] = node
        return node

    def make_sequence(self, items: Sequence[InternalChild]) -> InternalNode:
        """ Create collection """
        items: Tuple[InternalChild] = tuple(items)
        if node := self.__collections.get(items):
            return node

        node = InternalNode(SyntaxSequence, items)
        self.__collections[items] = node
        return node


InternalChild = InternalToken | InternalNode | None


# === Module -----------------------------------------------------------------------------------------------------------
class ModuleSyntax(SyntaxNode):
    """ Represents a module syntax node """

    imports: Sequence[ImportSyntax]
    """ The tree's imports """

    members: Sequence[FunctionSyntax]
    """ The tree's members """

    token_eof: SyntaxToken
    """ The `end of file` token """


# === Qualified identifier ---------------------------------------------------------------------------------------------
class IdentifierSyntax(SyntaxNode):
    @property
    @abc.abstractmethod
    def full_name(self) -> str:
        """ The type's full identifier with dots """
        raise NotImplementedError


class SimpleIdentifierSyntax(IdentifierSyntax):
    """ Represents a simple identifier syntax node """

    token_name: SyntaxToken
    """ The name token """

    @property
    def name(self) -> str:
        """ The type's identifier """
        return self.token_name.value

    @property
    def full_name(self) -> str:
        """ The type's full identifier with dots """
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location


class QualifiedIdentifierSyntax(IdentifierSyntax):
    """ Represents a qualified identifier syntax node """

    head: IdentifierSyntax
    """ The head identifier """

    token_dot: SyntaxToken
    """ The `.` token """

    token_name: SyntaxToken
    """ The identifier token """

    @property
    def name(self) -> str:
        """ The type's identifier """
        return self.token_name.value

    @property
    def full_name(self) -> str:
        """ The type's full identifier with dots """
        return '.'.join((self.head.full_name, self.name))

    @property
    def location(self) -> Location:
        return self.head.location + self.token_name.location


# === Imports ----------------------------------------------------------------------------------------------------------
class ImportSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract base for any import nodes. """


class ModuleImportSyntax(ImportSyntax):
    """ Represents a module import node """

    token_import: SyntaxToken
    """ The `import` token """

    modules: Sequence[ModuleAliasSyntax]
    """ The sequence of import's modules """

    token_newline: SyntaxToken
    """ The new line token """

    @property
    def location(self) -> Location:
        return self.token_import.location


class AllImportSyntax(ImportSyntax):
    """ Represents a from import node """

    token_from: SyntaxToken
    """ The `from` token """

    module: IdentifierSyntax
    """ The import's module """

    token_import: SyntaxToken
    """ The `import` token """

    token_star: SyntaxToken
    """ The `*` token """

    token_newline: SyntaxToken
    """ The new line token """

    @property
    def location(self) -> Location:
        return self.token_from.location


class FromImportSyntax(ImportSyntax):
    """ Represents a from import node """
    token_from: SyntaxToken
    """ The `from` token """
    module: str
    """ The import's module """

    token_import: SyntaxToken
    """ The `import` token """

    aliases: Sequence[AliasSyntax]
    """ The sequence of import's modules """

    token_newline: SyntaxToken
    """ The new line token """

    @property
    def location(self) -> Location:
        return self.token_from.location


class AliasSyntax(SyntaxNode):
    """ Represents a import alias """

    token_name: SyntaxToken
    """ The alias' name """

    token_as: SyntaxToken | None
    """ The optional 'as' token """

    token_alias: SyntaxToken | None
    """ The optional alias token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class SeparatedAliasSyntax(SyntaxNode):
    """ Represents a separated alias syntax """

    alias: AliasSyntax
    """ The alias """

    token_comma: SyntaxToken | None
    """ The optional trailing `,` token """

    @property
    def location(self) -> Location:
        return self.alias.location


class ModuleAliasSyntax(SyntaxNode):
    """ Represents a import alias """

    qualified_name: IdentifierSyntax
    """ The alias' name """

    token_as: SyntaxToken | None
    """ The optional 'as' token """

    token_alias: SyntaxToken | None
    """ The optional alias token """

    @property
    def location(self) -> Location:
        return self.qualified_name.location


class SeparatedModuleAliasSyntax(SyntaxNode):
    """ Represents a separated module alias syntax """

    alias: ModuleAliasSyntax
    """ The module alias """

    token_comma: SyntaxToken | None
    """ The optional trailing `,` token """

    @property
    def location(self) -> Location:
        return self.alias.location


# === Members ----------------------------------------------------------------------------------------------------------
class MemberSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract base for any member syntax nodes """


# === Effects ----------------------------------------------------------------------------------------------------------
class EffectSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract base for any effect syntax nodes """


class SeparatedEffectSyntax(SyntaxNode):
    """ Represents a separated effect syntax node """

    effect: EffectSyntax
    """ The separated effect """

    token_comma: SyntaxToken | None
    """ The optional `,` token """

    @property
    def location(self) -> Location:
        return self.effect.location


class IdentifierEffectSyntax(EffectSyntax):
    """ Represents a identifier effect node """

    qualified_name: IdentifierSyntax
    """ The name token """

    @property
    def location(self) -> Location:
        return self.qualified_name.location


class ParameterizedEffectSyntax(EffectSyntax):
    """ Represents a parameterized effect syntax node """

    qualified_name: IdentifierSyntax
    """ The parameterized effect's name """

    token_left_square: SyntaxToken | None
    """ The optional `[` token """

    arguments: Sequence[SeparatedEffectSyntax]
    """ The parameterized effect's arguments """

    token_right_square: SyntaxToken | None
    """ The optional `]` token """

    @property
    def location(self) -> Location:
        return self.qualified_name.location


# === Types ------------------------------------------------------------------------------------------------------------
class TypeSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract base for any type syntax nodes """


class SeparatedTypeSyntax(SyntaxNode):
    """ Represents a separated type syntax node """

    type: TypeSyntax
    """ The separated type """

    token_comma: SyntaxToken | None
    """ The optional `,` token """

    @property
    def location(self) -> Location:
        return self.type.location


class ExpansionTypeSyntax(TypeSyntax):
    """ Represents a identifier type node """

    type: TypeSyntax
    """ The expanded type """

    token_ellipsis: SyntaxToken
    """ The `...` token """

    @property
    def location(self) -> Location:
        return self.type.location


class IdentifierTypeSyntax(TypeSyntax):
    """ Represents a identifier type node """

    qualified_name: IdentifierSyntax
    """ The name token """

    @property
    def location(self) -> Location:
        return self.qualified_name.location


class ParameterizedTypeSyntax(TypeSyntax):
    """ Represents a parameterized type syntax node """

    qualified_name: IdentifierSyntax
    """ The parameterized type's name """

    token_left_square: SyntaxToken | None
    """ The optional `[` token """

    arguments: Sequence[SeparatedTypeSyntax]
    """ The parameterized type's arguments """

    token_right_square: SyntaxToken | None
    """ The optional `]` token """

    @property
    def location(self) -> Location:
        return self.qualified_name.location


class TupleTypeSyntax(TypeSyntax):
    """ Represents a tuple type syntax """
    token_left_parenthesis: SyntaxToken | None
    """ The `(` token """

    elements: Sequence[SeparatedTypeSyntax]
    """ The element types """

    token_right_parenthesis: SyntaxToken | None
    """ The `)` token """


class ArrayTypeSyntax(TypeSyntax):
    """ Represents a array type node """
    token_left_square: SyntaxToken
    """ The `[` token """

    element_type: TypeSyntax
    """ The element type """

    token_right_square: SyntaxToken
    """ The `]` token """


class DictionaryTypeSyntax(TypeSyntax):
    """ Represents a dictionary type node """
    token_left_curly: SyntaxToken
    """ The `{` token """

    key_type: TypeSyntax
    """ The key type """

    token_colon: SyntaxToken
    """ The `:` token """

    value_type: TypeSyntax
    """ The value type """

    token_right_curly: SyntaxToken
    """ The `}` token """


class SetTypeSyntax(TypeSyntax):
    """ Represents a set type node """
    token_left_curly: SyntaxToken
    """ The `{` token """

    element_type: TypeSyntax
    """ The key type """

    token_right_curly: SyntaxToken
    """ The `}` token """


class FunctionTypeSyntax(TypeSyntax):
    """ Represents a function type node """

    token_left_parenthesis: SyntaxToken
    """ The `(` token """

    parameters: Sequence[TypeSyntax]
    """ The function type's parameters """

    token_right_parenthesis: SyntaxToken
    """ The `)` token """

    token_arrow: SyntaxToken
    """ The optional `->` token """

    returns: TypeSyntax
    """ The function type's return type"""


class UnionTypeSyntax(TypeSyntax):
    """ Represents a union type node """

    left_type: TypeSyntax
    """ The left value """

    token_vertical_line: SyntaxToken
    """ The `|` token """

    right_type: TypeSyntax
    """ The right value """


class OptionalTypeSyntax(TypeSyntax):
    """ Represents an optional type node """

    element_type: TypeSyntax
    """ The element type """

    token_question: SyntaxToken
    """ The `?` token """


class PointerTypeSyntax(TypeSyntax):
    """ Represents an optional type node """

    element_type: TypeSyntax
    """ The element type """

    token_star: SyntaxToken
    """ The `*` token """


class DoublePointerTypeSyntax(TypeSyntax):
    """ Represents an optional type node """

    element_type: TypeSyntax
    """ The element type """

    token_double_star: SyntaxToken
    """ The `**` token """


# === Generic parameters -----------------------------------------------------------------------------------------------
class GenericParameterSyntax(SyntaxNode, abc.ABC):
    """ Represents a generic parameter syntax node """


class TypeGenericParameterSyntax(GenericParameterSyntax):
    """ Represents a type generic parameter syntax node """

    token_name: SyntaxToken
    """ The name token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class VariadicGenericParameterSyntax(GenericParameterSyntax):
    """ Represents a variadic generic parameter syntax node """

    token_name: SyntaxToken
    """ The name token """

    token_ellipsis: SyntaxToken
    """ The `...` token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class ValueGenericParameterSyntax(GenericParameterSyntax):
    """ Represents a value generic parameter syntax node """

    token_name: SyntaxToken
    """ The name token """

    token_colon: SyntaxToken
    """ The `:` token """

    type: TypeSyntax
    """ The value's type """

    @property
    def location(self) -> Location:
        return self.token_name.location


class EffectGenericParameterSyntax(GenericParameterSyntax):
    """ Represents a effect generic parameter syntax node """

    token_effect: SyntaxToken
    """ The `effect` token """

    token_name: SyntaxToken
    """ The name token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class SeparatedGenericParameterSyntax(SyntaxNode):
    """ Represents a separated generic parameter syntax node """

    parameter: GenericParameterSyntax
    """ The separated generic parameter """

    token_comma: SyntaxToken | None
    """ The optional `,` token """

    @property
    def location(self) -> Location:
        return self.parameter.location


# === Type members -----------------------------------------------------------------------------------------------------
class PassMemberSyntax(MemberSyntax):
    """ Represents a pass member syntax node """

    token_pass: SyntaxToken
    """ The `pass` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_pass.location


class FieldSyntax(MemberSyntax):
    """ Represents a field member syntax node """

    token_name: SyntaxToken
    """ The field's name """

    token_colon: SyntaxToken
    """ The `:` token """

    type: TypeSyntax
    """ The field's type """

    token_equal: SyntaxToken | None
    """ The `=` token """

    default_value: ExpressionSyntax | None
    """ The field's default value """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class EnumerationConstantMemberSyntax(MemberSyntax):
    """ Represents a enumeration constant member syntax node """

    token_name: SyntaxToken
    """ The field's target """

    token_equal: SyntaxToken
    """ The `=` token """

    default_value: ExpressionSyntax
    """ The field's default value """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_name.location


# === Type declarations ------------------------------------------------------------------------------------------------
class StructSyntax(MemberSyntax):
    """ Represents a struct syntax node """

    token_struct: SyntaxToken
    """ The `struct` token """

    token_name: SyntaxToken
    """ The name token """

    token_left_square: SyntaxToken | None
    """ The optional `[` token """

    generic_parameters: Sequence[SeparatedGenericParameterSyntax]
    """ The struct's generic parameters """

    token_right_square: SyntaxToken | None
    """ The optional `]` token """

    token_left_parenthesis: SyntaxToken | None
    """ The `(` token """

    bases: Sequence[SeparatedTypeSyntax]
    """ The struct's bases """

    token_right_parenthesis: SyntaxToken | None
    """ The `)` token """

    token_colon: SyntaxToken
    """ The `:` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    token_indent: SyntaxToken
    """ The `indent` token """

    members: Sequence[MemberSyntax]
    """ The sequence of nested members """

    token_dedent: SyntaxToken
    """ The `dedent` token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class InterfaceSyntax(MemberSyntax):
    """ Represents an interface syntax node """

    token_interface: SyntaxToken
    """ The `interface` token """

    token_name: SyntaxToken
    """ The name token """

    token_left_square: SyntaxToken | None
    """ The optional `[` token """

    generic_parameters: Sequence[SeparatedGenericParameterSyntax]
    """ The interface's generic parameters """

    token_right_square: SyntaxToken | None
    """ The optional `]` token """

    token_left_parenthesis: SyntaxToken | None
    """ The `(` token """

    bases: Sequence[SeparatedTypeSyntax]
    """ The interface's bases """

    token_right_parenthesis: SyntaxToken | None
    """ The `)` token """

    token_colon: SyntaxToken
    """ The `:` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    token_indent: SyntaxToken
    """ The `indent` token """

    members: Sequence[MemberSyntax]
    """ The sequence of nested members """

    token_dedent: SyntaxToken
    """ The `dedent` token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class ClassSyntax(MemberSyntax):
    """ Represents a class syntax node """

    token_class: SyntaxToken
    """ The `class` token """

    token_name: SyntaxToken
    """ The name token """

    token_left_square: SyntaxToken | None
    """ The optional `[` token """

    generic_parameters: Sequence[SeparatedGenericParameterSyntax]
    """ The class's generic parameters """

    token_right_square: SyntaxToken | None
    """ The optional `]` token """

    token_left_parenthesis: SyntaxToken | None
    """ The `(` token """

    bases: Sequence[SeparatedTypeSyntax]
    """ The class's bases """

    token_right_parenthesis: SyntaxToken | None
    """ The `)` token """

    token_colon: SyntaxToken
    """ The `:` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    token_indent: SyntaxToken
    """ The `indent` token """

    members: Sequence[MemberSyntax]
    """ The sequence of nested members """

    token_dedent: SyntaxToken
    """ The `dedent` token """

    @property
    def location(self) -> Location:
        return self.token_name.location


class EnumerationSyntax(MemberSyntax):
    """ Represents a enumeration syntax node """

    token_enum: SyntaxToken
    """ The `enum` token """

    token_name: SyntaxToken
    """ The name token """

    token_left_square: SyntaxToken | None
    """ The optional `[` token """

    generic_parameters: Sequence[SeparatedGenericParameterSyntax]
    """ The enumeration's generic parameters """

    token_right_square: SyntaxToken | None
    """ The optional `]` token """

    token_left_parenthesis: SyntaxToken | None
    """ The `(` token """

    bases: Sequence[SeparatedTypeSyntax]
    """ The enumeration's bases """

    token_right_parenthesis: SyntaxToken | None
    """ The `)` token """

    token_colon: SyntaxToken
    """ The `:` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    token_indent: SyntaxToken
    """ The `indent` token """

    members: Sequence[MemberSyntax]
    """ The sequence of nested members """

    token_dedent: SyntaxToken
    """ The `dedent` token """

    @property
    def location(self) -> Location:
        return self.token_name.location


# === Functions --------------------------------------------------------------------------------------------------------
class FunctionSyntax(MemberSyntax):
    """ Represents a function declaration syntax node """

    token_def: SyntaxToken
    """ The `def` token """

    token_name: SyntaxToken
    """ The name token """

    token_left_square: SyntaxToken | None
    """ The optional `[` token """

    generic_parameters: Sequence[SeparatedGenericParameterSyntax]
    """ The function's generic parameters """

    token_right_square: SyntaxToken | None
    """ The optional `]` token """

    token_left_parenthesis: SyntaxToken
    """ The `(` token """

    parameters: Sequence[SeparatedParameterSyntax]
    """ The function's parameters """

    token_right_parenthesis: SyntaxToken
    """ The `)` token """

    effects: Sequence[EffectSyntax]
    """ The function's effects """

    token_arrow: SyntaxToken | None
    """ The optional `->` token """

    returns: TypeSyntax | None
    """ The optional function's return type """

    token_colon: SyntaxToken
    """ The `:` token """

    statement: StatementSyntax
    """ The function's body """

    @property
    def name(self) -> str:
        """ The function's name """
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location


class ParameterSyntax(SyntaxNode):
    """ Represents a parameter syntax node """

    token_name: SyntaxToken
    """ The name token """

    token_colon: SyntaxToken
    """ The `:` token """

    type: TypeSyntax
    """ The 'parameter's type """

    token_ellipsis: SyntaxToken | None
    """ The `...` token """

    @property
    def name(self) -> str:
        """ The parameter's name """
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location


class SeparatedParameterSyntax(SyntaxNode):
    """ Represents a separated parameter syntax node """

    parameter: ParameterSyntax
    """ The separated parameter """

    token_comma: SyntaxToken | None
    """ The optional `,` token """

    @property
    def location(self) -> Location:
        return self.parameter.location


# === Statements -------------------------------------------------------------------------------------------------------
class StatementSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract base for any statement syntax nodes """


class EllipsisStatementSyntax(StatementSyntax):
    """ Represents a ellipsis statement syntax node, e.g. node that marked empty function body """

    token_ellipsis: SyntaxToken
    """ The `...` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_ellipsis.location


class BlockStatementSyntax(StatementSyntax):
    """ Represents a block statement syntax node, e.g. node that indent nested statements """

    token_newline: SyntaxToken
    """ The `new line` token """

    token_indent: SyntaxToken
    """ The `indent` token """

    statements: Sequence[StatementSyntax]
    """ The sequence of nested statements """

    token_dedent: SyntaxToken
    """ The `dedent` token """

    @property
    def location(self) -> Location:
        if not self.statements:
            return self.token_indent.location
        return self.statements[0].location + self.statements[-1].location


class PassStatementSyntax(StatementSyntax):
    """ Represents a pass statement syntax node """

    token_pass: SyntaxToken
    """ The `pass` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_pass.location


class ReturnStatementSyntax(StatementSyntax):
    """ Represents a return statement syntax node """

    token_return: SyntaxToken
    """ The `return` token """

    value: ExpressionSyntax | None
    """ The optional return value """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_return.location


class RaiseStatementSyntax(StatementSyntax):
    """ Represents a raise statement syntax node """

    token_raise: SyntaxToken
    """ The `raise` token """

    exception: ExpressionSyntax | None
    """ The optional raised exception """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_raise.location


class ContinueStatementSyntax(StatementSyntax):
    """ Represents a continue statement syntax node """

    token_continue: SyntaxToken
    """ The `continue` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_continue.location


class BreakStatementSyntax(StatementSyntax):
    """ Represents a break statement syntax node """
    token_break: SyntaxToken
    """ The `break` token """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_break.location


class WhileStatementSyntax(StatementSyntax):
    """ Represents a while statement syntax node """

    token_while: SyntaxToken
    """ The `while` token """

    condition: ExpressionSyntax
    """ The condition value """

    token_colon: SyntaxToken
    """ The `:` token """

    statement: StatementSyntax
    """ The nested statement """

    @property
    def location(self) -> Location:
        return self.token_while.location


class IfStatementSyntax(StatementSyntax):
    """ Represents a if statement syntax node """

    token_if: SyntaxToken
    """ The `if` or `elif` token """

    condition: ExpressionSyntax
    """ The condition value """

    token_colon: SyntaxToken
    """ The `:` token """

    then_statement: StatementSyntax
    """ The then statement """

    else_statement: StatementSyntax | None
    """ The else statement """

    @property
    def location(self) -> Location:
        return self.token_if.location


class ElseStatementSyntax(StatementSyntax):
    """ Represents a else statement syntax node """

    token_else: SyntaxToken
    """ The `else` token """

    token_colon: SyntaxToken
    """ The `:` token """

    statement: StatementSyntax
    """ The nested statement """

    @property
    def location(self) -> Location:
        return self.token_else.location


class TryExceptStatementSyntax(StatementSyntax):
    """ Represents a try/except statement syntax node """

    token_try: SyntaxToken
    """ The `try` token """

    token_colon: SyntaxToken
    """ The `:` token """

    statement: StatementSyntax
    """ The try statement """

    except_statements: Sequence[ExceptStatementSyntax]
    """ The sequence of except statements """

    else_statement: ElseStatementSyntax | None
    """ The optional else statement """

    finally_statement: FinallyStatementSyntax | None
    """ The optional finally statement """

    @property
    def location(self) -> Location:
        return self.token_try.location


class FinallyStatementSyntax(StatementSyntax):
    """ Represents a finally statement syntax node """

    token_finally: SyntaxToken
    """ The `finally` token """

    token_colon: SyntaxToken
    """ The `:` token """

    statement: StatementSyntax
    """ The nested statement """

    @property
    def location(self) -> Location:
        return self.token_finally.location


class TryFinallyStatementSyntax(StatementSyntax):
    """ Represents a try/finally statement syntax node """

    token_try: SyntaxToken
    """ The `try` token """

    token_colon: SyntaxToken
    """ The `:` token """

    statement: StatementSyntax
    """ The try statement """

    finally_statement: FinallyStatementSyntax
    """ The optional finally statement """

    @property
    def location(self) -> Location:
        return self.token_try.location


class ExceptStatementSyntax(SyntaxNode):
    """ Represents a except syntax node """

    token_except: SyntaxToken
    """ The `except` token """

    exception_type: TypeSyntax | None
    """ The optional type of exception """

    token_as: SyntaxToken | None
    """ The `as` token """

    exception_name: str | None
    """ The optional name of exception """

    token_colon: SyntaxToken | None
    """ The `:` token """

    statement: StatementSyntax
    """ The except's statement """

    @property
    def location(self) -> Location:
        return self.token_except.location


class ExpressionStatementSyntax(StatementSyntax):
    """ Represents a expression statement syntax node """

    value: ExpressionSyntax
    """ The value of expression statement """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.value.location


class AssignmentStatementSyntax(StatementSyntax):
    """ Represents an assignment statement syntax node """

    target: TargetSyntax
    """ The assignment's target """

    token_equal: SyntaxToken
    """ The `=` token """

    source: ExpressionSyntax
    """ The assignment's source """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def location(self) -> Location:
        return self.token_equal.location


class VariableStatementSyntax(StatementSyntax):
    """ Represents an variable statement syntax node """

    token_name: SyntaxToken
    """ The variable's target """

    token_colon: SyntaxToken
    """ The `:` token """

    type: TypeSyntax
    """ The variable's type """

    token_equal: SyntaxToken | None
    """ The `=` token """

    default_value: ExpressionSyntax | None
    """ The variable's default value """

    token_newline: SyntaxToken
    """ The `new line` token """

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location


# === Targets ----------------------------------------------------------------------------------------------------------

class TargetSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract base for all target syntax nodes """


class TupleTargetSyntax(TargetSyntax):
    """ Represents a tuple target syntax node """
    token_left_parenthesis: SyntaxToken
    """ The `(` token """

    elements: Sequence[SeparatedTargetSyntax]
    """ The nested tuple elements """

    token_right_parenthesis: SyntaxToken
    """ The `)` token """


class SeparatedTargetSyntax(SyntaxNode):
    """ Represents a separated target syntax node """

    value: TargetSyntax
    """ The separated value """

    token_comma: SyntaxToken | None
    """ The optional `,` token """

    @property
    def location(self) -> Location:
        return self.value.location


# === Expressions ------------------------------------------------------------------------------------------------------
class ExpressionSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract base for any expression syntax nodes """


class SeparatedExpressionSyntax(ExpressionSyntax):
    """ Represents a separated expression syntax node """

    value: ExpressionSyntax
    """ The separated value """

    token_comma: SyntaxToken | None
    """ The optional `,` token """

    @property
    def location(self) -> Location:
        return self.value.location


class ExpansionExpressionSyntax(ExpressionSyntax):
    """ Represents a identifier syntax node """

    expression: ExpressionSyntax
    """ The expanded expression """

    token_ellipsis: SyntaxToken
    """ The `...` token """

    @property
    def location(self) -> Location:
        return self.expression.location


class EllipsisExpressionSyntax(ExpressionSyntax):
    """ Represents a ellipsis expression syntax node """

    token_ellipsis: SyntaxToken
    """ The `...` token """


class IntegerExpressionSyntax(ExpressionSyntax):
    """ Represents a integer literal syntax node """

    token_value: SyntaxToken
    """ The integer token """

    @property
    def value(self) -> int:
        return ast.literal_eval(self.token_value.value)


class FloatExpressionSyntax(ExpressionSyntax):
    """ Represents a float literal syntax node """

    token_value: SyntaxToken
    """ The float token """

    @property
    def value(self) -> float:
        return ast.literal_eval(self.token_value.value)


class StringExpressionSyntax(ExpressionSyntax):
    """ Represents a string literal syntax node """

    token_value: SyntaxToken
    """ The string token """

    @property
    def value(self) -> str:
        return ast.literal_eval(self.token_value.value)


class IdentifierExpressionSyntax(TargetSyntax, ExpressionSyntax):
    """ Represents a identifier syntax node """

    token_name: SyntaxToken
    """ The identifier token """

    @property
    def name(self) -> str:
        """ The variable's identifier """
        return self.token_name.value


class CallExpressionSyntax(ExpressionSyntax):
    """ Represents a call expression syntax node """

    functor: ExpressionSyntax
    """ The functor """

    token_left_parenthesis: SyntaxToken
    """ The `(` token """

    arguments: Sequence[SeparatedExpressionSyntax]
    """ The sequence of call's arguments """

    token_right_parenthesis: SyntaxToken
    """ The `)` token """


class SubscriptExpressionSyntax(TargetSyntax, ExpressionSyntax):
    """ Represents a subscript expression syntax node """

    instance: ExpressionSyntax
    """ The instance """

    token_left_square: SyntaxToken
    """ The `[` token """

    arguments: Sequence[ExpressionSyntax]
    """ The sequence of subscript's arguments"""

    token_right_square: SyntaxToken
    """ The `]` token """


class AttributeExpressionSyntax(TargetSyntax, ExpressionSyntax):
    """ Represents a attribute syntax node """

    instance: ExpressionSyntax
    """ The instance """

    token_dot: SyntaxToken
    """ The `.` token """

    token_name: SyntaxToken
    """ The `name` token """

    @property
    def location(self) -> Location:
        return self.token_dot.location


class ParenthesisExpressionSyntax(ExpressionSyntax):
    """ Represents a parenthesis expression syntax node """

    token_left_parenthesis: SyntaxToken
    """ The `(` token """

    value: ExpressionSyntax
    """ The nested value """

    token_right_parenthesis: SyntaxToken
    """ The `)` token """


class TupleExpressionSyntax(ExpressionSyntax):
    """ Represents a tuple expression syntax node """
    token_left_parenthesis: SyntaxToken
    """ The `(` token """

    elements: Sequence[SeparatedExpressionSyntax]
    """ The nested tuple elements """

    token_right_parenthesis: SyntaxToken
    """ The `)` token """


class TupleComprehensionSyntax(ExpressionSyntax):
    """ Represents a tuple comprehension syntax """

    token_left_parenthesis: SyntaxToken
    """ The `(` token """

    element: ExpressionSyntax
    """ The tuple comprehension's element """

    comprehension: ForComprehensionSyntax
    """ The tuple comprehension's for """

    token_right_parenthesis: SyntaxToken
    """ The `)` token """


class ListExpressionSyntax(ExpressionSyntax):
    """ Represents a list expression syntax node """
    token_left_square: SyntaxToken
    """ The `[` token """

    elements: Sequence[SeparatedExpressionSyntax]
    """ The nested list elements """

    token_right_square: SyntaxToken
    """ The `]` token """


class ListComprehensionSyntax(ExpressionSyntax):
    """ Represents a list comprehension syntax """
    token_left_square: SyntaxToken
    """ The `[` token """

    element: ExpressionSyntax
    """ The list comprehension's element """

    comprehension: ForComprehensionSyntax
    """ The list comprehension """

    token_right_square: SyntaxToken
    """ The `]` token """


class SetExpressionSyntax(ExpressionSyntax):
    """ Represents a set expression syntax node """
    token_left_curly: SyntaxToken
    """ The `{` token """

    elements: Sequence[SeparatedExpressionSyntax]
    """ The nested set elements """

    token_right_curly: SyntaxToken
    """ The `}` token """


class SetComprehensionSyntax(ExpressionSyntax):
    """ Represents a set comprehension syntax """
    token_left_curly: SyntaxToken
    """ The `{` token """

    element: ExpressionSyntax
    """ The set comprehension's element """

    comprehension: ForComprehensionSyntax
    """ The set comprehension's for """

    token_right_curly: SyntaxToken
    """ The `}` token """


class DictionaryExpressionSyntax(ExpressionSyntax):
    """ Represents a dictionary expression syntax node """
    token_left_curly: SyntaxToken
    """ The `{` token """

    elements: Sequence[DictionaryElementSyntax]
    """ The nested dictionary elements """

    token_right_curly: SyntaxToken
    """ The `}` token """


class DictionaryElementSyntax(SyntaxNode):
    """ Represents a dictionary element syntax node """

    key: ExpressionSyntax
    """ The dictionary' key """

    token_colon: SyntaxToken
    """ The `:` token """

    value: ExpressionSyntax
    """ The dictionary' value """

    token_comma: SyntaxToken | None
    """ The optional `,` token """


class DictionaryComprehensionSyntax(ExpressionSyntax):
    """ Represents a dictionary comprehension syntax """

    key: ExpressionSyntax
    """ The set comprehension's key """

    value: ExpressionSyntax
    """ The set comprehension's value """

    comprehension: ForComprehensionSyntax
    """ The dictionary comprehension's for """


class ComprehensionSyntax(SyntaxNode, abc.ABC):
    """ Represents an abstract comprehension syntax """


class ForComprehensionSyntax(ExpressionSyntax):
    """ Represents a for comprehension syntax """

    token_for: SyntaxToken
    """ The `for` token """

    target: TargetSyntax
    """ The for comprehension's target """

    token_in: SyntaxToken
    """ The `in` token """

    source: ExpressionSyntax
    """ The for comprehension's source """

    comprehension: ComprehensionSyntax | None
    """ The for comprehension's next iterator """

    @property
    def location(self) -> Location:
        return self.token_for.location


class IfComprehensionSyntax(ExpressionSyntax):
    """ Represents an if comprehension syntax """

    token_if: SyntaxToken
    """ The `if` token """

    condition: ExpressionSyntax
    """ The condition comprehension's condition """

    comprehension: ComprehensionSyntax | None
    """ The for comprehension's next iterator """

    @property
    def location(self) -> Location:
        return self.token_if.location


class UnaryExpressionSyntax(ExpressionSyntax):
    """ Represents an abstract base for unary expression node """

    token_operator: SyntaxToken
    """ The operator token """

    value: ExpressionSyntax
    """ The value """

    @property
    def location(self) -> Location:
        return self.token_operator.location


class BinaryExpressionSyntax(ExpressionSyntax):
    """ Represents an abstract base for binary expression node """

    left_value: ExpressionSyntax
    """ The left value """

    token_operator: SyntaxToken
    """ The operator token """

    right_value: ExpressionSyntax
    """ The right value """

    @property
    def location(self) -> Location:
        return self.token_operator.location


class PosExpressionSyntax(UnaryExpressionSyntax):
    """ Represents a positive expression node """


class NegExpressionSyntax(UnaryExpressionSyntax):
    """ Represents a negative expression node """


class NotExpressionSyntax(UnaryExpressionSyntax):
    """ Represents a negative expression node """


class InvertExpressionSyntax(UnaryExpressionSyntax):
    """ Represents a invert expression node """


class PowExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a power expression node """


class AddExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a addition expression node """


class SubExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a substract expression node """


class MulExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a multiplication expression node """


class DivExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a divide expression node """


class FloorDivExpressionSyntax(BinaryExpressionSyntax):
    """ Represents floor divide power expression node """


class ModExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a module expression node """


class RightShiftExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a right shift expression node """


class LeftShiftExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a left shift expression node """


class BitwiseAndExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a bitwise and expression node """


class BitwiseXorExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a bitwise xor expression node """


class BitwiseOrExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a bitwise or expression node """


class LogicalOrExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a logical or expression node """


class LogicalAndExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a logical and expression node """


class EqualExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a equal expression node """


class NotEqualExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a not equal expression node """


class GreatExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a great then expression node """


class GreatEqualExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a great equal expression node """


class LessExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a less then expression node """


class LessEqualExpressionSyntax(BinaryExpressionSyntax):
    """ Represents a less equal expression node """
