# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
from typing import TypeVar, Generic as PyGeneric, Type as PyType, MutableMapping, Any, Sequence, AbstractSet, Set, \
    Iterable

from koine.undefined import *

TContext = TypeVar('TContext', bound='Context')
TContainer = TypeVar('TContainer', bound='Container')
TMember = TypeVar('TMember', bound='Member')
TModule = TypeVar('TModule', bound='Module')
TType = TypeVar('TType', bound='Type')
TFunction = TypeVar('TFunction', bound='Function')
TInstruction = TypeVar('TInstruction', bound='Instruction')
TControl = TypeVar('TControl', bound='Control')
TGeneric = TypeVar('TGeneric', bound='Generic')
TParameter = TypeVar('TParameter', bound='GenericParameter')
TArgument = TypeVar('TArgument', bound='GenericArgument')

BUILTINS_MODULE = '__builtins__'


# === Core: context ----------------------------------------------------------------------------------------------------
class Context(PyGeneric[TModule], abc.ABC):
    """ Represents a symbol context, e.g. manager for all shared symbols. """

    def __init__(self, loader: Loader[TModule]):
        self.__loader = loader
        self.__registered_modules: Set[TModule] = set()
        self.__cached_types: MutableMapping[Any, Any] = {}

    @property
    def registered_modules(self) -> AbstractSet[TModule]:
        """
        Returns all modules that initialized in this context

        :return: Sequence of modules
        """
        return self.__registered_modules

    @property
    def builtins_module(self) -> TModule:
        """ The module that contains builtins members. """
        return self.load(BUILTINS_MODULE)

    def load(self, name: str) -> TModule:
        return self.__loader.load(name)

    def _register_module(self, module: TModule):
        self.__registered_modules.add(module)

    def _get_cached_type(self, class_type: PyType, arguments: Sequence[Any]) -> Any:
        return self.__cached_types.get((class_type, *arguments))

    def _set_cached_type(self, class_type: PyType, arguments: Sequence[Any], instance: Any):
        self.__cached_types[(class_type, *arguments)] = instance


# === Core: module loader ----------------------------------------------------------------------------------------------
class Loader(PyGeneric[TModule], abc.ABC):
    """ Represents a module loader. """

    def load(self, name: str) -> TModule:
        raise NotImplementedError


# === Core: symbols ----------------------------------------------------------------------------------------------------
class Symbol(PyGeneric[TContext], abc.ABC):
    @property
    @abc.abstractmethod
    def context(self) -> TContext:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reference(self) -> str:
        """ The symbol's reference """
        raise NotImplementedError

    @property
    def is_generic(self) -> bool:
        return False

    def __str__(self) -> str:
        return self.reference

    def __repr__(self) -> str:
        return str(self)


class Name(PyGeneric[TContext], Symbol[TContext], abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """ The symbol's name """
        raise NotImplementedError

    @property
    def reference(self) -> str:
        """ The symbol's reference """
        return self.name


# === Core: containers -------------------------------------------------------------------------------------------------
class Container(PyGeneric[TContext, TMember], Symbol[TContext], abc.ABC):
    __members = Undefined

    @property
    def members(self) -> Sequence[TMember]:
        self.__members = unwrap_undefined(self.__members, self._unwrap_members)
        return load_undefined(self.__members)

    @members.setter
    def members(self, value: Iterable[TMember]):
        self.__members = wrap_undefined(self.__members, value, self._unwrap_members)

    def _unwrap_members(self, members: Iterable[TMember]) -> Sequence[TMember]:
        members = tuple(members)
        for member in members:
            member.owner = self
        return members


# === Core: members ----------------------------------------------------------------------------------------------------
class Member(PyGeneric[TContext, TContainer], Name[TContext], abc.ABC):
    @property
    def context(self) -> TContext:
        return self.parent.context

    @property
    @abc.abstractmethod
    def parent(self) -> TContainer:
        raise NotImplementedError


# === Core: module -----------------------------------------------------------------------------------------------------
class Module(PyGeneric[TContext, TMember], Name[TContext], Container[TContext, TMember]):
    def __init__(self, context: TContext, name: str, filename: str):
        self.__context = context
        self.__name = name
        self.__filename = filename

    @property
    def context(self) -> TContext:
        return self.__context

    @property
    def name(self) -> str:
        return self.__name


# === Core: generics ---------------------------------------------------------------------------------------------------
class Generic(PyGeneric[TContext, TContainer, TGeneric, TParameter, TArgument], Member[TContext, TContainer], abc.ABC):
    """ Represents an interface for all generic symbols """

    __original = Undefined
    __type_parameters = Undefined
    __type_arguments = Undefined

    @property
    def reference(self) -> str:
        return self.generic_name

    @property
    def generic_name(self) -> str:
        """ The symbol's name with type arguments """
        if self.type_arguments:
            parameters = ', '.join(param.reference for param in self.type_arguments)
            return f'{self.name}[{parameters}]'
        return self.name

    @property
    def is_generic(self) -> bool:
        return bool(self.type_parameters)

    @property
    def original(self) -> TGeneric | None:
        """ The original definition of this generic symbol. """
        return self.__original or None

    @original.setter
    def original(self, value: TGeneric | None):
        """ Assign original definition of this generic symbol """
        self.__original = store_undefined(self.__original, value)

    @property
    def type_parameters(self) -> Sequence[TParameter]:
        """
        The type parameters that this generic symbol has.

        If this is a non-generic symbol, returns an empty sequence
        """
        return self.__type_parameters or ()

    @type_parameters.setter
    def type_parameters(self, value: Sequence[TParameter]):
        """ Assign the type parameters that this generic symbol has """
        self.__type_parameters = wrap_undefined(self.__type_parameters, value, self._unwrap_type_parameters)
        self.__type_arguments = store_undefined(self.__type_arguments, self.__type_parameters)
        self.__original = store_undefined(self.__original, None)

        # TODO: Cache generic type?

    @property
    def type_arguments(self) -> Sequence[TArgument]:
        """
        Returns the type arguments that have been substituted for the type parameters.

        If nothing has been substituted for a given type parameter, then the type parameter itself is considered the
        type argument.
        """

        self.__type_arguments = unwrap_undefined(self.__type_arguments, tuple)
        return self.__type_arguments or self.type_parameters

    @type_arguments.setter
    def type_arguments(self, value: Sequence[TArgument]):
        """
        Assign the type arguments that have been substituted for the type parameters.
        """
        if not self.original:
            raise RuntimeError('Can not set type argument before origin')

        self.__type_arguments = wrap_undefined(self.__type_arguments, value, tuple)
        self.__type_parameters = store_undefined(self.__type_parameters, ())

        # TODO: Cache generic type?

    def _unwrap_type_parameters(self, parameters: Iterable[TParameter]) -> Sequence[TParameter]:
        parameters = tuple(parameters)
        for param in parameters:
            param.declared_symbol = self
        return parameters

    def instantiate(self, type_arguments: Sequence[TArgument]) -> TGeneric:
        """ Instantiate this generic symbol with type arguments that substituted for the type parameters """
        if not self.is_generic:
            raise ValueError(f'Can not instantiate symbol: non generic {self}')

        if len(self.type_parameters) != len(type_arguments):
            raise ValueError(f'Can not instantiate symbol: mismatch count of type arguments')

        return self._instantiate(type_arguments)

    @abc.abstractmethod
    def _instantiate(self, type_arguments: Sequence[TArgument]) -> TGeneric:
        raise NotImplementedError


class GenericArgument(PyGeneric[TContext], Symbol[TContext], abc.ABC):
    """ Represents an interface for all symbols that can be used as generic argument """


class GenericParameter(PyGeneric[TContext], Name[TContext], GenericArgument[TContext], abc.ABC):
    """ Represents an interface for all symbols that can be used as generic parameter """
    __declared_symbol = Undefined

    @property
    def declared_symbol(self) -> TGeneric | None:
        return load_undefined(self.__declared_symbol)

    @declared_symbol.setter
    def declared_symbol(self, value: TGeneric):
        self.__declared_symbol = store_undefined(self.__declared_symbol, value)

    @property
    def is_generic(self) -> bool:
        return False
