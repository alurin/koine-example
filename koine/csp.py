# Copyright (C) 2023 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
from typing import Sequence, Any, Type as PyType, Generic as PyGeneric, AbstractSet, Iterable, TypeVar

from koine.undefined import *


# === CPS: context -----------------------------------------------------------------------------------------------------
class CPSContext(abc.ABC):
    """
    Represents a symbol context, e.g. manager for all shared symbols.
    """

    def __init__(self):
        self.__registered_modules = set()
        self.__cached_types = {}
        self.__cached_constants = {}

    @property
    def registered_modules(self) -> AbstractSet[CPSModule]:
        """
        Returns all global and nested modules that contains in module

        :return: Sequence of modules
        """
        return self.__registered_modules

    def _register_module(self, module: CPSModule):
        self.__registered_modules.add(module)

    def _get_cached_type(self, class_type: PyType, arguments: Sequence[Any]) -> Any:
        return self.__cached_types.get((class_type, *arguments))

    def _set_cached_type(self, class_type: PyType, arguments: Sequence[Any], instance: Any):
        self.__cached_types[(class_type, *arguments)] = instance

    def _get_cached_constant(self, type: CPSType, value: Any) -> CPSConstant | None:
        return self.__cached_constants.get((type, value))

    def _set_cached_constant(self, type: CPSType, value: Any, instance: CPSConstant):
        self.__cached_constants[(type, value)] = instance


# === CPS: core --------------------------------------------------------------------------------------------------------
class CPSNode(abc.ABC):
    def __init__(self):
        self.__uses = set()

        for node in self.inputs:
            node.__uses.add(self)

    @property
    @abc.abstractmethod
    def context(self) -> CPSContext:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def inputs(self) -> AbstractSet[CPSNode]:
        raise NotImplementedError

    @property
    def uses(self) -> AbstractSet[CPSNode]:
        return self.__uses

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)


class CPSSymbol(CPSNode, abc.ABC):
    @property
    @abc.abstractmethod
    def reference(self) -> str:
        """ The symbol's reference """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.reference

    def __repr__(self) -> str:
        return str(self)


class CPSName(CPSSymbol, abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """ The symbol's name """
        raise NotImplementedError

    @property
    def reference(self) -> str:
        """ The symbol's reference """
        return self.name


# === CPS: container ---------------------------------------------------------------------------------------------------
class CPSContainer(CPSSymbol, abc.ABC):
    __members = Undefined

    @property
    def members(self) -> Sequence[CPSMember]:
        self.__members = unwrap_undefined(self.__members, self._unwrap_members)
        return load_undefined(self.__members)

    @members.setter
    def members(self, value: Iterable[CPSMember]):
        self.__members = wrap_undefined(self.__members, value, self._unwrap_members)

    def _unwrap_members(self, members: Iterable[CPSMember]) -> Sequence[CPSMember]:
        members = tuple(members)
        for member in members:
            member.owner = self
        return members


class CPSMember(CPSName, abc.ABC):
    @property
    def context(self) -> CPSContext:
        return self.parent.context

    @property
    @abc.abstractmethod
    def parent(self) -> CPSContainer:
        raise NotImplementedError


# === CPS: generics ----------------------------------------------------------------------------------------------------
TGeneric = TypeVar('TGeneric', bound='Generic')


class CPSGeneric(CPSName, PyGeneric[TGeneric], abc.ABC):
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
    def type_parameters(self) -> Sequence[CPSGenericParameter]:
        """
        The type parameters that this generic symbol has.

        If this is a non-generic symbol, returns an empty sequence
        """
        return self.__type_parameters or ()

    @type_parameters.setter
    def type_parameters(self, value: Sequence[CPSGenericParameter]):
        """ Assign the type parameters that this generic symbol has """
        self.__type_parameters = wrap_undefined(self.__type_parameters, value, self.unwrap_type_parameters)
        self.__type_arguments = store_undefined(self.__type_arguments, self.__type_parameters)
        self.__original = store_undefined(self.__original, None)

        # TODO: Cache generic type?

    @property
    def type_arguments(self) -> Sequence[CPSGenericArgument]:
        """
        Returns the type arguments that have been substituted for the type parameters.

        If nothing has been substituted for a given type parameter, then the type parameter itself is considered the
        type argument.
        """

        self.__type_arguments = unwrap_undefined(self.__type_arguments, tuple)
        return self.__type_arguments or self.type_parameters

    @type_arguments.setter
    def type_arguments(self, value: Sequence[CPSGenericArgument]):
        """
        Assign the type arguments that have been substituted for the type parameters.
        """
        if not self.original:
            raise RuntimeError('Can not set type argument before origin')

        self.__type_arguments = wrap_undefined(self.__type_arguments, value, tuple)
        self.__type_parameters = store_undefined(self.__type_parameters, ())

        # TODO: Cache generic type?

    def unwrap_type_parameters(self, parameters: Iterable[CPSGenericParameter]) -> Sequence[CPSGenericParameter]:
        parameters = tuple(parameters)
        for param in parameters:
            param.declared_symbol = self
        return parameters

    def instantiate(self, type_arguments: Sequence[CPSGenericArgument]) -> TGeneric:
        raise NotImplementedError

        # """ Instantiate this generic symbol with type arguments that substituted for the type parameters """
        # if not self.is_generic:
        #     raise ValueError(f'Can not instantiate symbol: non generic {self}')
        #
        # if len(self.type_parameters) != len(type_arguments):
        #     raise ValueError(f'Can not instantiate symbol: mismatch count of type arguments')
        #
        # mapping = RewriteMapper(module, zip(self.type_parameters, type_arguments))
        # if self.original:
        #     type_arguments = tuple(mapping.rewrite(param) for param in self.type_arguments)
        #     return self.original.instantiate(module, type_arguments)
        #
        # return mapping.instantiate(self)


class CPSGenericArgument(CPSSymbol, abc.ABC):
    """ Represents an interface for all symbols that can be used as generic argument """


class CPSGenericParameter(CPSName, CPSGenericArgument, abc.ABC):
    """ Represents an interface for all symbols that can be used as generic parameter """
    __declared_symbol = Undefined

    @property
    def declared_symbol(self) -> CPSGeneric | None:
        return load_undefined(self.__declared_symbol)

    @declared_symbol.setter
    def declared_symbol(self, value: CPSGeneric):
        self.__declared_symbol = store_undefined(self.__declared_symbol, value)

    @property
    def is_generic(self) -> bool:
        return False
