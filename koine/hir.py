# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
from functools import cached_property
from typing import Sequence, Any, Type as PyType, AbstractSet, Iterator

import more_itertools

from koine import core
from koine import priorities


# === HIR: context -----------------------------------------------------------------------------------------------------
class HIRContext(core.Context['HIRModule']):
    """
    Represents a symbol context, e.g. manager for all shared symbols.
    """

    def __init__(self, loader: core.Loader[HIRModule]):
        super().__init__(loader)

        self.__cached_constants = {}

    def _get_cached_constant(self, type: HIRType, value: Any) -> HIRConstant | None:
        return self.__cached_constants.get((type, value))

    def _set_cached_constant(self, type: HIRType, value: Any, instance: HIRConstant):
        self.__cached_constants[(type, value)] = instance


# === HIR: core --------------------------------------------------------------------------------------------------------
class HIRNode(abc.ABC):
    @property
    @abc.abstractmethod
    def context(self) -> HIRContext:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)


class HIRSymbol(core.Symbol[HIRContext], abc.ABC):
    pass


class HIRName(core.Name[HIRContext], HIRSymbol, abc.ABC):
    pass


# === HIR: container ---------------------------------------------------------------------------------------------------
class HIRContainer(core.Container[HIRContext, 'HIRMember'], HIRSymbol, abc.ABC):
    pass


class HIRMember(core.Member[HIRContext, HIRContainer], abc.ABC):
    pass


# === HIR: generics ----------------------------------------------------------------------------------------------------
class HIRGenericArgument(core.GenericArgument[HIRContext], HIRSymbol, abc.ABC):
    """ Represents an interface for all symbols that can be used as generic argument """


class HIRGenericParameter(core.GenericParameter[HIRContext], HIRGenericArgument, abc.ABC):
    """ Represents an interface for all symbols that can be used as generic parameter """


# === HIR: module ------------------------------------------------------------------------------------------------------
class HIRModule(HIRName, HIRContainer):
    def __init__(self, context: HIRContext, name: str, filename: str):
        self.__context = context
        self.__name = name
        self.__filename = filename

        self.__registered_effects = set()
        self.__registered_types = set()
        self.__registered_functions = set()

        super().__init__()

        # noinspection PyProtectedMember
        self.__context._register_module(self)

    @property
    def context(self) -> HIRContext:
        return self.__context

    @property
    def name(self) -> str:
        return self.__name

    @property
    def filename(self) -> str:
        """ The path to source code for this module """
        return self.__filename

    @property
    def registered_effects(self) -> AbstractSet[HIREffect]:
        """
        Returns all global effects contains in module

        :return: Sequence of types
        """
        return self.__registered_effects

    @property
    def registered_types(self) -> AbstractSet[HIRType]:
        """
        Returns all global types contains in module

        :return: Sequence of types
        """
        return self.__registered_types

    @property
    def registered_functions(self) -> AbstractSet[HIRFunction]:
        """
        Returns all global functions contains in module

        :return: Sequence of functions
        """
        return self.__registered_functions

    def _register_effect(self, effect: HIREffect):
        self.__registered_effects.add(effect)

    def _register_type(self, type: HIRType):
        self.__registered_types.add(type)

    def _register_function(self, func: HIRFunction):
        self.__registered_functions.add(func)


# === HIR: effects -----------------------------------------------------------------------------------------------------
class HIREffect(core.Generic[HIRContext, HIRContainer, 'HIREffect', HIRGenericParameter, HIRGenericArgument]):
    def __init__(self, context: HIRContext, name: str):
        self.__context = context
        self.__name = name

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__context

    @property
    def name(self) -> str:
        return self.__name


class HIROperator(HIRName):
    def __init__(self, name: str, parameters: Sequence[HIRVariable], returns: HIRType):
        self.__name = name
        self.__parameters = parameters
        self.__returns = returns

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__returns.context

    @property
    def name(self) -> str:
        return self.__name

    @property
    def parameters(self) -> Sequence[HIRVariable]:
        return self.__parameters

    @property
    def returns(self) -> HIRType:
        return self.__returns

    def __str__(self) -> str:
        parameters = ', '.join(str(param) for param in self.parameters)
        return f'{self.name}({parameters}) -> {self.returns.reference}'


# === HIR: types -------------------------------------------------------------------------------------------------------
class HIRType(HIRGenericArgument, abc.ABC):
    def __or__(self, other: HIRType) -> HIRType | HIRUnionType:
        elements = simplify_set({self, other}, HIRUnionType)
        if len(elements) == 1:
            return more_itertools.first(elements)
        return HIRUnionType(self.context, elements)


class HIRGenericType(HIRGenericParameter, HIRType):
    def __init__(self, context: HIRContext, name: str):
        self.__context = context
        self.__name = name

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__context

    @property
    def name(self) -> str:
        return self.__name


class _HIRTypeConstructorArgumentMeta(abc.ABCMeta):
    """ Metaclass for type that cached in context """

    # noinspection PyProtectedMember
    def __call__(cls, context: HIRContext, *args):
        # use cached version of type
        if instance := context._get_cached_type(cls, args):
            return instance

        # generate cached version of type
        instance = super(_HIRTypeConstructorArgumentMeta, cls).__call__(context, *args)
        context._set_cached_type(cls, args, instance)
        return instance


class _HIRTypeConstructorSingleMeta(abc.ABCMeta):
    """ Metaclass for union type that cached in context """

    # noinspection PyProtectedMember
    def __call__(cls, element_type: HIRType):
        context = element_type.context

        # use cached version of type
        if instance := context._get_cached_type(cls, (element_type,)):
            return instance

        # generate cached version of type
        instance = super(_HIRTypeConstructorSingleMeta, cls).__call__(element_type)
        context._set_cached_type(cls, (element_type,), instance)
        return instance


class _HIRTypeConstructorSequenceMeta(abc.ABCMeta):
    # noinspection PyMethodOverriding
    def __call__(cls, context, elements: Sequence[HIRType]):
        elements = tuple(elements)

        # use cached version of type
        if instance := context._get_cached_type(cls, (elements,)):
            return instance

        # generate cached version of type
        instance = super(_HIRTypeConstructorSequenceMeta, cls).__call__(context, elements)
        context._set_cached_type(cls, (elements,), instance)
        return instance


class _HIRTypeConstructorSetMeta(abc.ABCMeta):
    # noinspection PyMethodOverriding
    def __call__(cls, context: HIRContext, elements: AbstractSet[HIRType]) -> HIRType:
        elements = simplify_set(elements, HIRUnionType)

        match len(elements):
            case 0:
                raise RuntimeError('Can not create union type from empty elements')
            case 1:
                return more_itertools.first(elements)
            case _:
                # use cached version of type
                if instance := context._get_cached_type(cls, (elements,)):
                    return instance

                # generate cached version of type
                instance = super(_HIRTypeConstructorSetMeta, cls).__call__(context, elements)
                context._set_cached_type(cls, (elements,), instance)
                return instance


class _HIRTypeConstructorFunctionMeta(abc.ABCMeta):
    """ Metaclass for union type that cached in context """

    # noinspection PyProtectedMember
    def __call__(cls, parameters: Sequence[HIRType], returns: HIRType, effects: AbstractSet[HIREffect]):
        context = returns.context
        parameters = tuple(parameters)
        effects = frozenset(parameters)

        # use cached version of type
        if instance := context._get_cached_type(cls, (parameters, returns, effects)):
            return instance

        # generate cached version of type
        instance = super(_HIRTypeConstructorFunctionMeta, cls).__call__(parameters, returns, effects)
        context._set_cached_type(cls, (parameters, returns, effects), instance)
        return instance


class HIRPrimitiveType(HIRType, abc.ABC, metaclass=_HIRTypeConstructorArgumentMeta):
    def __init__(self, context: HIRContext):
        self.__context = context

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__context

    def __eq__(self, other: HIRType) -> bool:
        return isinstance(other, type(self))

    def __hash__(self) -> int:
        return hash(type(self))


# `*` - is dynamic type
class HIRDynamicType(HIRPrimitiveType):
    @property
    def reference(self) -> str:
        return 'any'


# `⊤` - is a supertype of all types
class HIRAnyType(HIRPrimitiveType):
    @property
    def reference(self) -> str:
        return '⊤'


# `⊥` - is a subtype of all types
class HIRVoidType(HIRPrimitiveType):
    @property
    def reference(self) -> str:
        return '⊥'


class HIRBooleanType(HIRPrimitiveType):
    @property
    def reference(self) -> str:
        return 'bool'


class HIRIntegerType(HIRPrimitiveType):
    @property
    def reference(self) -> str:
        return 'int'


class HIRFloatType(HIRPrimitiveType):
    @property
    def reference(self) -> str:
        return 'float'


class HIRStringType(HIRPrimitiveType):
    @property
    def reference(self) -> str:
        return 'str'


class HIRFunctionType(HIRType, metaclass=_HIRTypeConstructorFunctionMeta):
    def __init__(self, parameters: Sequence[HIRType], returns: HIRType, effects: AbstractSet[HIREffect]):
        self.__parameters = tuple(parameters)
        self.__effects = frozenset(effects)
        self.__returns = returns

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__returns.context

    @cached_property
    def reference(self) -> str:
        parameters = ', '.join(param.reference for param in self.parameters)
        effects = ' '.join(effect.reference for effect in self.__effects)
        return f'({parameters}) {effects}{" " if effects else ""}-> {self.returns.reference}'

    @property
    def parameters(self) -> Sequence[HIRType]:
        """ The parameters of function type """
        return self.__parameters

    @property
    def effects(self) -> AbstractSet[HIREffect]:
        return self.__effects

    @property
    def returns(self) -> HIRType:
        """ The result type of function type """
        return self.__returns

    def __hash__(self) -> int:
        return hash(tuple((type(self), self.__returns, *self.__parameters, *self.__effects)))

    def __eq__(self, other: HIRType) -> bool:
        return self is other \
            or isinstance(other, HIRFunctionType) \
            and other.__returns == self.__returns \
            and self.__parameters == other.__parameters \
            and self.__effects == other.__effects


class HIRArrayType(HIRType, metaclass=_HIRTypeConstructorSingleMeta):
    def __init__(self, element_type: HIRType):
        self.__element_type = element_type

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__element_type.context

    @property
    def reference(self) -> str:
        return f'[{type_parenthesis(self, self.element_type)}]'

    @property
    def element_type(self) -> HIRType:
        return self.__element_type


class HIRTupleType(HIRType, metaclass=_HIRTypeConstructorSequenceMeta):
    def __init__(self, context: HIRContext, elements: Sequence[HIRType]):
        self.__context = context
        self.__elements = elements

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__context

    @property
    def reference(self) -> str:
        return ', '.join(type_parenthesis(self, element) for element in self.elements)

    @property
    def elements(self) -> Sequence[HIRType]:
        return self.__elements


class HIRUnionType(HIRType, metaclass=_HIRTypeConstructorSetMeta):
    def __init__(self, context: HIRContext, elements: AbstractSet[HIRType]):
        self.__context = context
        self.__elements = elements

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__context

    @property
    def elements(self) -> AbstractSet[HIRType]:
        return self.__elements

    @property
    def reference(self) -> str:
        return ' | '.join(type_parenthesis(self, element) for element in self.elements)


# === HIR: values ------------------------------------------------------------------------------------------------------
class HIRValue(HIRSymbol, abc.ABC):
    @property
    def context(self) -> HIRContext:
        return self.type.context

    @property
    @abc.abstractmethod
    def type(self) -> HIRType:
        raise NotImplementedError


class HIRConstant(HIRValue):
    def __init__(self, type: HIRType, value: Any):
        self.__type = type
        self.__value = value

        super().__init__()

    @property
    def reference(self) -> str:
        return repr(self.__value)

    @property
    def type(self) -> HIRType:
        return self.__type

    @property
    def value(self) -> Any:
        return self.__value


class HIRFunction(HIRName, HIRValue):
    def __init__(self, name: str, parameters: Sequence[HIRVariable], returns: HIRType, effects: Sequence[HIREffect]):
        self.__name = name
        self.__parameters = tuple(parameters)
        self.__effects = frozenset(effects)
        self.__returns = returns

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__returns.context

    @property
    def name(self):
        return self.__name

    @property
    def parameters(self) -> Sequence[HIRVariable]:
        return self.__parameters

    @property
    def effects(self) -> AbstractSet[HIREffect]:
        return self.__effects

    @property
    def returns(self) -> HIRType:
        return self.__returns

    @cached_property
    def type(self) -> HIRFunctionType:
        parameters = [param.type for param in self.__parameters]
        effects = {effect for effect in self.__effects}
        return HIRFunctionType(parameters, self.__returns, effects)

    def __str__(self) -> str:
        parameters = ', '.join(str(param) for param in self.parameters)
        effects = ' '.join(effect.reference for effect in self.__effects)
        return f'{self.name}({parameters}) {effects}{" " if effects else ""}-> {self.returns.reference}'


class HIRVariable(HIRName, HIRValue):
    def __init__(self, name: str, type: HIRType):
        self.__name = name
        self.__type = type

        super().__init__()

    @property
    def name(self) -> str:
        return self.__name

    @property
    def type(self) -> HIRType:
        return self.__type

    def __str__(self) -> str:
        return f'{self.__name}: {self.__type.reference}'


class HIRTuple(HIRValue):
    def __init__(self, context: HIRContext, elements: Sequence[HIRValue]):
        self.__context = context
        self.__elements = elements

        super().__init__()

    @property
    def reference(self) -> str:
        elements = ', '.join(element.reference for element in self.__elements)
        return f'({elements})'

    @property
    def elements(self) -> Sequence[HIRValue]:
        return self.__elements

    @cached_property
    def type(self) -> HIRTupleType:
        elements = [element.type for element in self.__elements]
        return HIRTupleType(self.__context, elements)


# === HIR: statements and expressions ----------------------------------------------------------------------------------
class HIRInstruction(HIRNode, abc.ABC):
    pass


class HIRCallInst(HIRInstruction, HIRValue):
    def __init__(self, functor: HIRFunction, arguments: Sequence[HIRValue]):
        self.__returns = functor.returns
        self.__functor = functor
        self.__arguments = arguments

        super().__init__()

    @property
    def type(self) -> HIRType:
        return self.__returns

    @property
    def reference(self) -> str:
        arguments = ', '.join(arg.reference for arg in self.__arguments)
        return f'{self.__functor.reference}({arguments})'

    @property
    def functor(self):
        return self.__functor

    @property
    def arguments(self):
        return self.__arguments


class HIRApplyInst(HIRInstruction, HIRValue):
    def __init__(self, operator: HIROperator, arguments: Sequence):
        self.__returns = operator.returns
        self.__operator = operator
        self.__arguments = tuple(arguments)

        super().__init__()

    @property
    def reference(self) -> str:
        arguments = ', '.join(arg.reference for arg in self.__arguments)
        return f'{self.__operator.reference}({arguments})'

    @property
    def operator(self) -> HIROperator:
        return self.__operator

    @property
    def arguments(self) -> Sequence[HIRValue]:
        return self.__arguments

    @property
    def type(self) -> HIRType:
        return self.__returns


class HIRReturnInst(HIRInstruction):
    def __init__(self, value: HIRValue):
        self.__value = value

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__value.context

    def __str__(self) -> str:
        return f'return {self.__value.reference}'


class HIREvalInst(HIRInstruction):
    def __init__(self, value: HIRValue):
        self.__value = value

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__value.context

    def __str__(self) -> str:
        return f'eval {self.__value.reference}'


class HIRRegionInst(HIRInstruction):
    def __init__(self, instructions: Sequence[HIRInstruction]):
        assert len(instructions) > 0

        self.__context = instructions[0].context
        self.__instructions = instructions
        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__context

    @property
    def instructions(self) -> Sequence[HIRInstruction]:
        return self.__instructions

    def __str__(self) -> str:
        return '\n'.join(str(inst) for inst in self.__instructions)


class HIRBreakInst(HIRInstruction):
    def __init__(self, condition: HIRValue, then_block: HIRInstruction, else_block: HIRInstruction | None):
        self.__condition = condition
        self.__then_block = then_block
        self.__else_block = else_block

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__condition.context

    @property
    def condition(self) -> HIRValue:
        return self.__condition

    @property
    def then_block(self) -> HIRInstruction:
        return self.__then_block

    @property
    def else_block(self) -> HIRInstruction | None:
        return self.__else_block

    def __str__(self) -> str:
        return f'break {self.__condition.reference} ... ...'


class HIRLoopInst(HIRInstruction):
    def __init__(self, condition: HIRValue, block: HIRInstruction):
        self.__condition = condition
        self.__block = block

        super().__init__()

    @property
    def context(self) -> HIRContext:
        return self.__condition.context

    @property
    def condition(self) -> HIRValue:
        return self.__condition

    @property
    def block(self) -> HIRInstruction:
        return self.__block

    def __str__(self) -> str:
        return f'break {self.__condition.reference} ... ...'


# === HIR: set helpers -------------------------------------------------------------------------------------------------
def expand_set(elements: AbstractSet[HIRType], kind: PyType[HIRType], expander) -> Iterator[HIRType]:
    for element in elements:
        if isinstance(element, kind):
            # noinspection PyUnresolvedReferences
            yield from expander(element)
        else:
            yield element


def simplify_set(elements: AbstractSet[HIRType], type_kind: PyType[HIRType]) -> AbstractSet[HIRType]:
    # expand and unique set
    return frozenset(expand_set(elements, type_kind, lambda e: e.elements))


# === HIR: priorities --------------------------------------------------------------------------------------------------
def get_type_priority(symbol: HIRType) -> int:
    match symbol:
        case HIRTupleType():
            return priorities.BINDING_MAX
        case HIRUnionType():
            return priorities.BITWISE_OR_BINDING
        case _:
            return priorities.BINDING_MIN


def type_parenthesis(parent: HIRType, element: HIRType) -> str:
    parent_rhs = get_type_priority(parent)
    element_rhs = get_type_priority(element)

    if element_rhs >= parent_rhs:
        return f'({element.reference})'
    return element.reference
