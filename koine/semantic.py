# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import functools
from typing import TypeAlias, Tuple, cast, Mapping, Callable

import multimethod
from pyrsistent import pmap

from koine import syntax
from koine.exceptions import SemanticError
from koine.hir import *
from koine.lazyness import LazyMapping
from koine.parser import fetch_syntax_tree
from koine.source import fetch_source_text
from koine.strings import quote_identifier as _
from koine.syntax import SyntaxTree

TypeSyntax: TypeAlias = syntax.TypeSyntax \
                        | syntax.SeparatedTypeSyntax \
                        | syntax.UsingDeclarationSyntax \
                        | syntax.TypeDeclarationSyntax

EffectSyntax: TypeAlias = syntax.EffectSyntax \
                          | syntax.SeparatedEffectSyntax \
                          | syntax.EffectDeclarationSyntax

GenericParameterSyntax: TypeAlias = syntax.GenericParameterSyntax \
                                    | syntax.SeparatedGenericParameterSyntax


class Scope(abc.ABC):
    pass


class SemanticModel:
    def __init__(self, tree: SyntaxTree):
        self.__tree = tree
        self.__context = HIRContext()
        self.__environments = LazyMapping(functools.partial(annotate_node_environment, self))

        def evaluate_in_environ(evaluate):
            return lambda n: evaluate(self.fetch_environment(n), n)

        self.__symbols = LazyMapping(functools.partial(annotate_node_symbol, self))
        self.__types = LazyMapping(evaluate_in_environ(synthesize_type))
        self.__effects = LazyMapping(evaluate_in_environ(synthesize_effect))
        self.__functions = LazyMapping(evaluate_in_environ(synthesize_function))
        self.__parameters = LazyMapping(evaluate_in_environ(synthesize_parameter))
        self.__expressions = LazyMapping(evaluate_in_environ(synthesize_expression))
        self.__operators = LazyMapping(evaluate_in_environ(synthesize_operator))
        self.__type_parameters = LazyMapping(evaluate_in_environ(synthesize_type_parameter))

    @property
    def tree(self) -> SyntaxTree:
        return self.__tree

    @property
    def context(self) -> HIRContext:
        return self.__context

    @property
    def root(self) -> syntax.ModuleSyntax:
        return self.__tree.root

    def fetch_environment(self, node: syntax.SyntaxNode) -> AbstractEnvironment:
        return self.__environments[node]

    def fetch_symbol(self, node: syntax.SyntaxNode) -> HIRSymbol | None:
        return self.__symbols[node]

    def fetch_type(self, node: TypeSyntax) -> HIRType:
        return self.__types[node]

    def fetch_type_parameter(self, node: GenericParameterSyntax) -> HIRGenericParameter:
        return self.__type_parameters[node]

    def fetch_effect(self, node: EffectSyntax) -> HIREffect:
        return self.__effects[node]

    def fetch_function(self, node: syntax.FunctionDeclarationSyntax) -> HIRFunction:
        return self.__functions[node]

    def fetch_operator(self, node: syntax.OperatorDeclarationSyntax) -> HIROperator:
        return self.__operators[node]

    def fetch_parameter(self, node: syntax.SeparatedParameterSyntax | syntax.ParameterSyntax) -> HIRVariable:
        return self.__parameters[node]

    def fetch_expression(self, node: syntax.ExpressionSyntax) -> HIRSymbol:
        return self.__expressions[node]


class AbstractEnvironment(abc.ABC):
    @property
    @abc.abstractmethod
    def model(self) -> SemanticModel:
        raise NotImplementedError

    @cached_property
    def context(self) -> HIRContext:
        return self.model.context

    def fetch_symbol(self, node: syntax.SyntaxNode) -> HIRSymbol | None:
        return self.model.fetch_symbol(node)

    def fetch_lazy_symbol(self, node: syntax.SyntaxNode) -> Callable[[], HIRSymbol | None]:
        return lambda: self.fetch_symbol(node)

    def fetch_type(self, node: TypeSyntax) -> HIRType:
        return self.model.fetch_type(node)

    def fetch_type_parameter(self, node: GenericParameterSyntax) -> HIRGenericParameter:
        return self.model.fetch_type_parameter(node)

    def fetch_effect(self, node: EffectSyntax) -> HIREffect:
        return self.model.fetch_effect(node)

    def fetch_function(self, node: syntax.FunctionDeclarationSyntax) -> HIRFunction:
        return self.model.fetch_function(node)

    def fetch_operator(self, node: syntax.OperatorDeclarationSyntax) -> HIROperator:
        return self.model.fetch_operator(node)

    def fetch_parameter(self, node: syntax.SeparatedParameterSyntax | syntax.ParameterSyntax) -> HIRVariable:
        return self.model.fetch_parameter(node)

    @abc.abstractmethod
    def resolve(self, name: str) -> HIRSymbol | None:
        raise NotImplementedError

    def resolve_type(self, node: syntax.IdentifierSyntax) -> HIRType:
        return synthesize_identifier_type(self, node)


class ModuleEnvironment(AbstractEnvironment):
    def __init__(self, model: SemanticModel, node: syntax.ModuleSyntax):
        self.__model = model
        self.__node = node
        self.__scope = None

    @property
    def model(self) -> SemanticModel:
        return self.__model

    @property
    def node(self) -> syntax.ModuleSyntax:
        return self.__node

    def resolve(self, name: str) -> HIRSymbol | None:
        if self.__scope is None:
            self.__scope = annotate_node_scope(self, self.node)

        if resolver := self.__scope.get(name):
            return resolver()
        return None


class DeclarationEnvironment(AbstractEnvironment):
    def __init__(self, parent: AbstractEnvironment, node: syntax.DeclarationSyntax):
        self.__parent = parent
        self.__node = node
        self.__scope = None

    @property
    def model(self) -> SemanticModel:
        return self.__parent.model

    @property
    def node(self) -> syntax.DeclarationSyntax:
        return self.__node

    def resolve(self, name: str) -> HIRSymbol | None:
        if self.__scope is None:
            self.__scope = annotate_node_scope(self, self.node)

        if resolver := self.__scope.get(name):
            return resolver()
        return None


class FunctionEnvironment(AbstractEnvironment):
    def __init__(self, parent: AbstractEnvironment, node: syntax.FunctionDeclarationSyntax):
        self.__parent = parent
        self.__node = node
        self.__environments = {}
        self.__symbols = {}

    @cached_property
    def model(self) -> SemanticModel:
        return self.__parent.model

    @property
    def node(self) -> syntax.FunctionDeclarationSyntax:
        return self.__node

    @property
    def function(self) -> HIRFunction:
        return self.fetch_function(self.__node)

    @property
    def returns(self) -> HIRType:
        return self.function.returns

    def resolve(self, name: str) -> HIRSymbol | None:
        return self.__parent.resolve(name)

    def check(self):
        if self.__environments:
            return  # Already checked function body

        initial = {param.name: param for param in self.function.parameters}
        env = StatementEnvironment(self, initial=initial)  # TODO: propagate parameters and effects to environment
        env.check_stmt(self.node.statement)

    def get_environment(self, node: syntax.SyntaxNode) -> StatementEnvironment:
        self.check()  # TODO: Force there?
        return self.__environments.get(node)

    def check_stmt(self, env: StatementEnvironment, node: syntax.StatementSyntax) -> StatementEnvironment:
        return self.__execute_in_statement(env, node, check_statement_types)

    def check_bool(self, env: StatementEnvironment, node: syntax.ExpressionSyntax) -> SemanticValue:
        return self.__execute_in_expression(env, node, check_boolean_type)

    def check_type(self, env: StatementEnvironment, node: syntax.ExpressionSyntax, expected: HIRType) -> SemanticValue:
        return self.__execute_in_expression(env, node, check_expression_type, expected)

    def synthesize_value(self, env: StatementEnvironment, node: syntax.ExpressionSyntax) -> SemanticValue:
        return self.__execute_in_expression(env, node, synthesize_expression_value)

    def synthesize_symbol(self, env: StatementEnvironment, node: syntax.ExpressionSyntax) -> SemanticSymbol:
        return self.__execute_in_expression(env, node, synthesize_expression_symbol)

    def __execute_in_statement(self, env: StatementEnvironment, node: syntax.StatementSyntax, annotator, *args):
        if node in self.__environments:
            raise SemanticError(node.location, 'Statement is already type checked')

        self.__environments[node] = env  # temporary environment for type checking
        env = annotator(env, node, *args)
        self.__environments[node] = env
        return env

    def __execute_in_expression(self, env: StatementEnvironment, node: syntax.ExpressionSyntax, annotator, *args):
        if node in self.__symbols:
            raise SemanticError(node.location, 'Expression is already type checked')

        self.__environments[node] = env  # temporary environment for type checking
        true_env, false_env, symbol = annotator(env, node, *args)
        self.__symbols[node] = symbol
        self.__environments[node] = true_env
        return true_env, false_env, symbol


class StatementEnvironment(AbstractEnvironment):
    def __init__(self,
                 parent: FunctionEnvironment,
                 predecessors: Sequence[StatementEnvironment] = None,
                 initial: Mapping[str, HIRVariable] = None):
        self.__parent = parent
        self.__predecessors = tuple(predecessors or ())
        self.__variables = pmap(initial or ())

    @cached_property
    def model(self) -> SemanticModel:
        return self.__parent.model

    @property
    def parent(self) -> FunctionEnvironment:
        return self.__parent

    @property
    def predecessors(self) -> Sequence[StatementEnvironment]:
        return self.__predecessors

    @property
    def function(self) -> HIRFunction:
        return self.__parent.function

    @property
    def returns(self) -> HIRType:
        return self.__parent.returns

    def check_stmt(self, node: syntax.StatementSyntax) -> StatementEnvironment:
        return self.__parent.check_stmt(self, node)

    def check_bool(self, node: syntax.ExpressionSyntax) -> SemanticValue:
        return self.__parent.check_bool(self, node)

    def check_type(self, node: syntax.ExpressionSyntax, expected: HIRType) -> SemanticValue:
        return self.__parent.check_type(self, node, expected)

    def synthesize_value(self, node: syntax.ExpressionSyntax) -> SemanticValue:
        return self.__parent.synthesize_value(self, node)

    def synthesize_symbol(self, node: syntax.ExpressionSyntax) -> SemanticSymbol:
        return self.__parent.synthesize_symbol(self, node)

    def declare(self, name: str, type: HIRType) -> Tuple[StatementEnvironment, HIRVariable]:
        env = StatementEnvironment(self.__parent, (self,))
        var = HIRVariable(name, type)
        env.__variables = self.__variables.set(name, var)
        return env, var

    def resolve(self, name: str) -> HIRSymbol | None:
        match name:
            case 'True':
                return HIRConstant(HIRBooleanType(self.context), True)
            case 'False':
                return HIRConstant(HIRBooleanType(self.context), False)

        if var := self.__variables.get(name):
            return var
        return self.__parent.resolve(name)

    def __eq__(self, other: StatementEnvironment) -> bool:
        raise NotImplementedError

    def __or__(self, other: StatementEnvironment) -> StatementEnvironment:
        env = StatementEnvironment(self.__parent, [self, other])

        for name, lhs in self.__variables.items():
            if rhs := other.__variables.get(name):
                env.__variables = env.__variables.set(name, HIRVariable(name, lhs.type | rhs.type))

        return env


SemanticSymbol: TypeAlias = Tuple[StatementEnvironment, StatementEnvironment, HIRSymbol]
SemanticValue: TypeAlias = Tuple[StatementEnvironment, StatementEnvironment, HIRValue]


def fetch_semantic_model(filename: str) -> SemanticModel:
    source = fetch_source_text(filename)
    tree = fetch_syntax_tree(source)
    model = SemanticModel(tree)

    for child in tree.root.members:
        if isinstance(child, syntax.DeclarationSyntax):
            symbol = model.fetch_symbol(child)
            print(symbol)
            if isinstance(symbol, HIRContainer):
                for member in symbol.members:
                    print('  :>', member)

    return model


# === Node environments ------------------------------------------------------------------------------------------------
@multimethod.multimethod
def annotate_node_environment(model: SemanticModel, node: syntax.SyntaxNode) -> AbstractEnvironment:
    return model.fetch_environment(node.parent)


@multimethod.multimethod
def annotate_node_environment(model: SemanticModel, node: syntax.ModuleSyntax) -> AbstractEnvironment:
    return ModuleEnvironment(model, node)


@multimethod.multimethod
def annotate_node_environment(model: SemanticModel, node: syntax.UsingDeclarationSyntax) -> AbstractEnvironment:
    return DeclarationEnvironment(model.fetch_environment(node.parent), node)


@multimethod.multimethod
def annotate_node_environment(model: SemanticModel, node: syntax.TypeDeclarationSyntax) -> AbstractEnvironment:
    return DeclarationEnvironment(model.fetch_environment(node.parent), node)


@multimethod.multimethod
def annotate_node_environment(model: SemanticModel, node: syntax.EffectDeclarationSyntax) -> AbstractEnvironment:
    return DeclarationEnvironment(model.fetch_environment(node.parent), node)


@multimethod.multimethod
def annotate_node_environment(model: SemanticModel, node: syntax.FunctionDeclarationSyntax) -> AbstractEnvironment:
    return FunctionEnvironment(model.fetch_environment(node.parent), node)


@multimethod.multimethod
def annotate_node_environment(model: SemanticModel, node: syntax.StatementSyntax) -> AbstractEnvironment:
    # Force typing check
    parent = node.parent
    while not isinstance(parent, syntax.FunctionDeclarationSyntax):
        parent = parent.parent

    func_env = cast(FunctionEnvironment, model.fetch_environment(parent))
    return func_env.get_environment(node)


# === Node environments ------------------------------------------------------------------------------------------------
@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.SyntaxNode) -> HIRSymbol | None:
    raise SemanticError(node.location, 'Not implemented symbol annotation')


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.DocumentationMemberSyntax) -> None:
    return None


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.GenericParameterSyntax) -> None:
    return model.fetch_type_parameter(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.SeparatedGenericParameterSyntax) -> HIRGenericParameter:
    return model.fetch_type_parameter(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.TypeSyntax) -> HIRType:
    return model.fetch_type(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.SeparatedTypeSyntax) -> HIRType:
    return model.fetch_type(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.TypeDeclarationSyntax) -> HIRType:
    return model.fetch_type(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.UsingDeclarationSyntax) -> HIRType:
    return model.fetch_type(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.EffectSyntax) -> HIREffect:
    return model.fetch_effect(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.SeparatedEffectSyntax) -> HIREffect:
    return model.fetch_effect(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.EffectDeclarationSyntax) -> HIREffect:
    return model.fetch_effect(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.ParameterSyntax) -> HIRVariable:
    return model.fetch_parameter(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.SeparatedParameterSyntax) -> HIRVariable:
    return model.fetch_parameter(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.FunctionDeclarationSyntax) -> HIRFunction:
    return model.fetch_function(node)


@multimethod.multimethod
def annotate_node_symbol(model: SemanticModel, node: syntax.OperatorDeclarationSyntax) -> HIROperator:
    return model.fetch_operator(node)


# === Node scopes ------------------------------------------------------------------------------------------------------
@multimethod.multimethod
def annotate_node_scope(env: AbstractEnvironment, node: syntax.SyntaxNode) -> Mapping[str, Callable[[], HIRSymbol]]:
    raise SemanticError(node.location, 'Not implemented lexical scope')


@multimethod.multimethod
def annotate_node_scope(env: AbstractEnvironment, node: syntax.ModuleSyntax) -> Mapping[str, Callable[[], HIRSymbol]]:
    return {
        member.name: env.fetch_lazy_symbol(member)
        for member in node.members
        if isinstance(member, syntax.DeclarationSyntax)
    }


@multimethod.multimethod
def annotate_node_scope(env: AbstractEnvironment, node: syntax.UsingDeclarationSyntax) \
        -> Mapping[str, Callable[[], HIRSymbol]]:
    return {param.name: env.fetch_lazy_symbol(param) for param in node.generic_parameters}


@multimethod.multimethod
def annotate_node_scope(env: AbstractEnvironment, node: syntax.EffectDeclarationSyntax) \
        -> Mapping[str, Callable[[], HIRSymbol]]:
    return {param.name: env.fetch_lazy_symbol(param) for param in node.generic_parameters}


# === Synthesize type parameters ---------------------------------------------------------------------------------------
@multimethod.multimethod
def synthesize_type_parameter(env: AbstractEnvironment, node: syntax.GenericParameterSyntax) -> HIRGenericParameter:
    raise SemanticError(node.location, 'Not implemented parameter annotation')


@multimethod.multimethod
def synthesize_type_parameter(env: AbstractEnvironment,
                              node: syntax.SeparatedGenericParameterSyntax) -> HIRGenericParameter:
    return env.fetch_type_parameter(node.parameter)


@multimethod.multimethod
def synthesize_type_parameter(env: AbstractEnvironment, node: syntax.TypeGenericParameterSyntax) -> HIRGenericParameter:
    return HIRGenericType(env.context, node.name)


# === Identifier types -------------------------------------------------------------------------------------------------
@multimethod.multimethod
def synthesize_identifier_type(env: AbstractEnvironment, node: syntax.IdentifierSyntax) -> HIRType:
    raise SemanticError(node.location, 'Not implemented type annotation for identifier')


@multimethod.multimethod
def synthesize_identifier_type(env: AbstractEnvironment, node: syntax.SimpleIdentifierSyntax) -> HIRType:
    if symbol := get_identifier_type(env, node.name):
        return symbol

    if symbol := env.resolve(node.name):
        if isinstance(symbol, HIRType):
            return symbol
        raise SemanticError(node.location, f'Can not use symbol {_(symbol.reference)} as value')
    raise SemanticError(node.location, f'Not found type {_(node.name)} in current scope')


def get_identifier_type(env: AbstractEnvironment, name: str) -> HIRType | None:
    match name:
        case 'int':
            return HIRIntegerType(env.context)
        case 'str':
            return HIRStringType(env.context)
        case 'float':
            return HIRFloatType(env.context)
        case 'bool':
            return HIRBooleanType(env.context)
        case 'nothing':
            return HIRVoidType(env.context)
        case 'any':
            return HIRDynamicType(env.context)


# === Synthesize type --------------------------------------------------------------------------------------------------
@multimethod.multimethod
def synthesize_type(env: AbstractEnvironment, node: syntax.TypeSyntax) -> HIRType:
    raise SemanticError(node.location, 'Not implemented type annotation for node')


@multimethod.multimethod
def synthesize_type(env: AbstractEnvironment, node: syntax.SeparatedTypeSyntax) -> HIRType:
    return env.fetch_type(node.type)


@multimethod.multimethod
def synthesize_type(env: AbstractEnvironment, node: syntax.ArrayTypeSyntax) -> HIRArrayType:
    element_type = env.fetch_type(node.element_type)
    return HIRArrayType(element_type)


@multimethod.multimethod
def synthesize_type(env: AbstractEnvironment, node: syntax.UnionTypeSyntax) -> HIRType:
    elements = {env.fetch_type(node.left_type), env.fetch_type(node.right_type)}
    return HIRUnionType(env.context, elements)


@multimethod.multimethod
def synthesize_type(env: AbstractEnvironment, node: syntax.IdentifierTypeSyntax) -> HIRType:
    return env.resolve_type(node.qualified_name)


@multimethod.multimethod
def synthesize_type(env: AbstractEnvironment, node: syntax.UsingDeclarationSyntax) -> HIRType:
    if node.generic_parameters:
        raise SemanticError(node.location, 'Not supported type aliases')
    if not isinstance(node.canonical_type, syntax.EllipsisTypeSyntax):
        raise SemanticError(node.location, 'Not supported type aliases')

    if symbol := get_identifier_type(env, node.name):
        return symbol

    raise SemanticError(node.location, 'Not supported type aliases')


# === Synthesize effect ------------------------------------------------------------------------------------------------
@multimethod.multimethod
def synthesize_effect(env: AbstractEnvironment, node: syntax.EffectSyntax) -> HIREffect:
    raise SemanticError(node.location, 'Not implemented effect annotation for node')


@multimethod.multimethod
def synthesize_effect(env: AbstractEnvironment, node: syntax.EffectDeclarationSyntax) -> HIREffect:
    effect = HIREffect(env.context, node.name)
    effect.type_parameters = [env.fetch_type_parameter(param) for param in node.generic_parameters]
    effect.members = lambda: [symbol for member in node.members if (symbol := env.fetch_symbol(member))]
    return effect


# === Synthesize function ----------------------------------------------------------------------------------------------
def synthesize_function(env: AbstractEnvironment, node: syntax.FunctionDeclarationSyntax) -> HIRFunction:
    parameters = [env.fetch_parameter(param) for param in node.parameters]
    returns = env.fetch_type(node.returns)
    effects = {env.fetch_effect(effect) for effect in node.effects}
    return HIRFunction(node.name, parameters, returns, effects)


def synthesize_operator(env: AbstractEnvironment, node: syntax.OperatorDeclarationSyntax) -> HIROperator:
    parameters = [env.fetch_parameter(param) for param in node.parameters]
    returns = env.fetch_type(node.returns)
    return HIROperator(node.name, parameters, returns)


@multimethod.multimethod
def synthesize_parameter(env: AbstractEnvironment, node: syntax.SeparatedParameterSyntax) -> HIRVariable:
    return env.fetch_parameter(node.parameter)


@multimethod.multimethod
def synthesize_parameter(env: AbstractEnvironment, node: syntax.ParameterSyntax) -> HIRVariable:
    param_type = env.fetch_type(node.type)
    return HIRVariable(node.name, param_type)


# === Synthesize expression --------------------------------------------------------------------------------------------
@multimethod.multimethod
def synthesize_expression(env: AbstractEnvironment, node: syntax.ExpressionSyntax) -> HIRSymbol:
    raise SemanticError(node.location, 'Not implemented symbol annotation for node')


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.StatementSyntax) -> StatementEnvironment:
    raise SemanticError(node.location, 'Not implemented type checking for statement')


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, _: syntax.EllipsisStatementSyntax) -> StatementEnvironment:
    return env  # Nothing checked


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, _: syntax.PassStatementSyntax) -> StatementEnvironment:
    return env  # Nothing checked


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.ElseStatementSyntax) -> StatementEnvironment:
    return env.check_stmt(node.statement)  # Check nested statement


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.FinallyStatementSyntax) -> StatementEnvironment:
    return env.check_stmt(node.statement)  # Check nested statement


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.BlockStatementSyntax) -> StatementEnvironment:
    for stmt in node.statements:
        env = env.check_stmt(stmt)
    return env


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.ExpressionStatementSyntax) -> StatementEnvironment:
    env, _, _ = env.synthesize_symbol(node.value)
    return env


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.ReturnStatementSyntax) -> StatementEnvironment:
    env, _, _ = env.check_type(node.value, env.returns)
    return env


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.IfStatementSyntax) -> StatementEnvironment:
    true_env, false_env, condition = env.check_bool(node.condition)

    true_env = true_env.check_stmt(node.then_statement)
    if node.else_statement:
        false_env = false_env.check_stmt(node.else_statement)

    return true_env | false_env


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.WhileStatementSyntax) -> StatementEnvironment:
    true_env, false_env, condition = env.check_bool(node.condition)

    # TODO: find closure for `while` body
    while True:
        body_env = true_env.check_stmt(node.statement)
        if body_env == true_env:
            break
        true_env = body_env | body_env  # new body environment

    return true_env | false_env


@multimethod.multimethod
def check_statement_types(env: StatementEnvironment, node: syntax.AssignmentStatementSyntax) -> StatementEnvironment:
    return check_assignment_types(env, node.target, node.source)


@multimethod.multimethod
def check_assignment_types(env: StatementEnvironment, target: syntax.TargetSyntax, source: syntax.ExpressionSyntax) \
        -> StatementEnvironment:
    raise SemanticError(target.location, 'Not implemented value assignment for target')


@multimethod.multimethod
def check_assignment_types(env: StatementEnvironment, target: syntax.IdentifierExpressionSyntax,
                           source: syntax.ExpressionSyntax) -> StatementEnvironment:
    if env.resolve(target.name):
        raise SemanticError(target.location, 'Reassignment is not implemented')

    env, _, value = env.synthesize_value(source)
    env, _ = env.declare(target.name, value.type)
    return env


@multimethod.multimethod
def check_boolean_type(env: StatementEnvironment, node: syntax.ExpressionSyntax) -> SemanticValue:
    true_env, false_env, value = env.synthesize_value(node)
    actual = value.type

    if isinstance(actual, HIRBooleanType):
        return true_env, false_env, value

    raise SemanticError(node.location,
                        f'Can not use value of type {_(actual.reference)} in boolean context')


@multimethod.multimethod
def check_expression_type(env: StatementEnvironment, node: syntax.ExpressionSyntax, expected: HIRType) -> SemanticValue:
    raise SemanticError(node.location, 'Not implemented type checking for statement')


@multimethod.multimethod
def check_expression_type(env: StatementEnvironment, node: syntax.ExpressionSyntax, expected: HIRType) -> SemanticValue:
    true_env, false_env, value = env.synthesize_value(node)
    actual = value.type

    if actual == expected:
        return true_env, false_env, value

    # TODO: There must propagation to integer/float type?

    raise SemanticError(node.location,
                        f'Can not convert value of type {_(actual.reference)} to {_(expected.reference)}')


@multimethod.multimethod
def synthesize_expression_value(env: StatementEnvironment, node: syntax.ExpressionSyntax) -> SemanticValue:
    true_env, false_env, symbol = env.synthesize_symbol(node)
    if isinstance(symbol, HIRValue):
        return true_env, false_env, symbol

    raise SemanticError(node.location, f'Can not use symbol {_(symbol.reference)} as value')


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.ExpressionSyntax) -> SemanticSymbol:
    raise SemanticError(node.location, 'Not implemented symbol annotation for expression')


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.SeparatedExpressionSyntax) -> SemanticSymbol:
    return env.synthesize_symbol(node.value)


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.IntegerExpressionSyntax) -> SemanticSymbol:
    return env, env, HIRConstant(HIRIntegerType(env.context), node.value)


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.FloatExpressionSyntax) -> SemanticSymbol:
    return env, env, HIRConstant(HIRFloatType(env.context), node.value)


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.StringExpressionSyntax) -> SemanticSymbol:
    return env, env, HIRConstant(HIRStringType(env.context), node.value)


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.IdentifierExpressionSyntax) -> SemanticSymbol:
    if symbol := env.resolve(node.name):
        return env, env, symbol

    raise SemanticError(node.location, f'Not found symbol {_(node.name)} in current scope')


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.TupleExpressionSyntax) -> SemanticSymbol:
    elements = []
    for node_element in node.elements:
        env, _, element = env.synthesize_value(node_element)
        elements.append(element)
    value = HIRTuple(env.context, elements)
    return env, env, value


@multimethod.multimethod
def synthesize_expression_symbol(env: StatementEnvironment, node: syntax.CallExpressionSyntax) -> SemanticSymbol:
    env, _, functor = env.synthesize_symbol(node.functor)

    if not isinstance(functor, HIRFunction):
        raise SemanticError(node.functor.location, 'Non callable object')

    expected_count = len(functor.parameters)
    actual_count = len(node.arguments)
    if expected_count != actual_count:
        raise SemanticError(node.location,
                            f'Mismatch count of arguments: required {expected_count}, but got {actual_count}')

    arguments = []
    for param, arg_node in zip(functor.parameters, node.arguments):
        env, _, arg = env.check_type(arg_node, param.type)
        arguments.append(arg)

    return env, env, HIRCall(functor, arguments)
