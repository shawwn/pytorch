from tools.codegen.model import (Argument, FunctionSchema, NativeFunction,
                                 BackendIndex,
                                 SelfArgument, TensorOptionsArguments, BaseTy)
from dataclasses import dataclass
from typing import Optional, Union, Sequence, TypeVar, List, Set, Dict
from enum import Enum

_T = TypeVar('_T')

# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occr.

SpecialArgName = Enum('SpecialArgName', (
    'possibly_redundant_memory_format',
))
ArgName = Union[str, SpecialArgName]

# This class shouldn't be created directly; instead, use/create one of the singletons below.
@dataclass(frozen=True)
class BaseCppType:
    ns: Optional[str]
    name: str

    def __str__(self) -> str:
        if self.ns is None or self.ns == '':
            return self.name
        return f"{self.ns}::{self.name}"

# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
byteT = BaseCppType('', 'uint8_t')
charT = BaseCppType('', 'int8_t')
shortT = BaseCppType('', 'int16_t')
# It would be more symmetric for this to be called intT, but it easy to mix
# this up with JIT int (which is int64_t in C++), so we intentionally don't
# define intT to make it obvious when you've stuffed it up
int32T = BaseCppType('', 'int32_t')
longT = BaseCppType('', 'int64_t')
halfT = BaseCppType('at', 'Half')
doubleT = BaseCppType('', 'double')
floatT = BaseCppType('', 'float')
complexHalfT = BaseCppType('c10', 'complex<c10::Half>')  # stuffing template param here is an abuse
complexFloatT = BaseCppType('c10', 'complex<float>')
complexDoubleT = BaseCppType('c10', 'complex<double>')
boolT = BaseCppType('', 'bool')
bfloat16T = BaseCppType('at', 'BFloat16')
voidT = BaseCppType('', 'void')
stringT = BaseCppType('c10', 'string_view')
generatorT = BaseCppType('at', 'Generator')
scalarTypeT = BaseCppType('at', 'ScalarType')
tensorT = BaseCppType('at', 'Tensor')
optionalTensorRefT = BaseCppType('at', 'OptionalTensorRef')
tensorListT = BaseCppType('at', 'TensorList')
dimnameT = BaseCppType('at', 'Dimname')
dimnameListT = BaseCppType('at', 'DimnameList')
layoutT = BaseCppType('at', 'Layout')
deviceT = BaseCppType('at', 'Device')
scalarT = BaseCppType('at', 'Scalar')
optionalScalarRefT = BaseCppType('at', 'OptionalScalarRef')
memoryFormatT = BaseCppType('at', 'MemoryFormat')
qschemeT = BaseCppType('at', 'QScheme')
storageT = BaseCppType('at', 'Storage')
streamT = BaseCppType('at', 'Stream')
intArrayRefT = BaseCppType('at', 'IntArrayRef')
tensorOptionsT = BaseCppType('at', 'TensorOptions')
typeAndSizeT = BaseCppType('torch::autograd::generated', 'TypeAndSize')
tensorGeometryT = BaseCppType('at', 'TensorGeometry')

BaseTypeToCppMapping: Dict[BaseTy, BaseCppType] = {
    BaseTy.int: longT,
    BaseTy.float: doubleT,
    BaseTy.bool: boolT,
    BaseTy.str: stringT,
    BaseTy.Generator: generatorT,
    BaseTy.ScalarType: scalarTypeT,
    BaseTy.Tensor: tensorT,
    BaseTy.Dimname: dimnameT,
    BaseTy.Layout: layoutT,
    BaseTy.Device: deviceT,
    BaseTy.Scalar: scalarT,
    BaseTy.MemoryFormat: memoryFormatT,
    BaseTy.QScheme: qschemeT,
    BaseTy.Storage: storageT,
    BaseTy.Stream: streamT,
}

# CTypes encode C++ type structure as needed for translation.

@dataclass(frozen=True)
class BaseCType:
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return str(self.type)

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def cpp_type_registration_declarations(self) -> str:
        return str(self.type).replace('at::', '')

    def remove_const_ref(self) -> 'CType':
        return self

@dataclass(frozen=True)
class ConstRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'const {self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        return f'const {self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class MutRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'{self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        return f'{self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class OptionalCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'c10::optional<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'c10::optional<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return OptionalCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ListCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'c10::List<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'c10::List<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return ListCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ArrayRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'at::ArrayRef<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'ArrayRef<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return ArrayRefCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class VectorCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::vector<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::vector<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return VectorCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ArrayCType:
    elem: 'CType'
    size: int

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::array<{self.elem.cpp_type()},{self.size}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::array<{self.elem.cpp_type_registration_declarations()},{self.size}>'

    def remove_const_ref(self) -> 'CType':
        return ArrayCType(self.elem.remove_const_ref(), self.size)

@dataclass(frozen=True)
class TupleCType:
    elems: List['CType']

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::tuple<{",".join([e.cpp_type() for e in self.elems])}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::tuple<{",".join([e.cpp_type_registration_declarations() for e in self.elems])}>'

    def remove_const_ref(self) -> 'CType':
        return TupleCType([e.remove_const_ref() for e in self.elems])

CType = Union[
    BaseCType,
    OptionalCType,
    ConstRefCType,
    MutRefCType,
    ListCType,
    ArrayRefCType,
    ArrayCType,
    VectorCType,
    TupleCType
]

# A NamedCType is short for Named C++ semantic type.  A NamedCType represents a C++ type, plus
# semantic information about what it represents.  For example, consider the
# argument "bool pin_memory"; its normal C++ type is "bool", but its C++
# semantic type also keeps track that this represents a "pin_memory"; you can't
# just use a random other boolean in a context where you need a "pin_memory"!
#

@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return self.type.cpp_type(strip_ref=strip_ref)

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def cpp_type_registration_declarations(self) -> str:
        return self.type.cpp_type_registration_declarations()

    def remove_const_ref(self) -> 'NamedCType':
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> 'NamedCType':
        return NamedCType(name, self.type)

# A binding represents any C++ binding site for a formal parameter.
# We don't distinguish between binding sites for different APIs;
# instead, all of the important distinctions are encoded in CType,
# which you can use to figure out if a given Binding is appropriate
# for use in another context.  (See tools.codegen.api.translate)

@dataclass(frozen=True)
class Binding:
    name: str
    nctype: NamedCType
    argument: Union[Argument, TensorOptionsArguments, SelfArgument]
    # TODO: maybe don't represent default here
    default: Optional[str] = None

    @property
    def type(self) -> str:
        return self.nctype.cpp_type()

    def no_default(self) -> 'Binding':
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    def decl(self, *, func_ptr_cast: bool = False) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"

        # casting only needs to know the type
        if func_ptr_cast:
            return f"{self.type}"
        else:
            return f"{self.type} {self.name}{mb_default}"

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def decl_registration_declarations(self) -> str:
        type_s = self.nctype.cpp_type_registration_declarations()
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{type_s} {self.name}{mb_default}"

    def defn(self) -> str:
        return f"{self.type} {self.name}"

# An Expr is a C++ expression.  It has a C++ string representing its syntax,
# as well as a CType saying what it provides.

@dataclass(frozen=True)
class Expr:
    expr: str
    type: NamedCType

# A CppSignature represents a single overload in the C++ API.  For
# any given function schema, there may be multiple CppSignatures
# corresponding to it, based on how we desugar to C++.  See also
# CppSignatureGroup.
@dataclass(frozen=True)
class CppSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Is this a C++ signature for a method, i.e. Tensor::my_op(...)?
    method: bool

    # Is this a faithful C++ signature (i.e. following the JIT schema) or a convenience API
    # (i.e. with a potential TensorOptions argument and out arguments in the front)
    faithful: bool

    # The set of C++ arguments which should not have defaults applied to them
    cpp_no_default_args: Set[str]

    # Is this a fallback C++ binding?  Fallback bindings are enabled by
    # manual_cpp_binding: True and are alternate, non-public API that
    # lets manual C++ binding implementors access the binding that would
    # have been automatically generated
    fallback_binding: bool = False

    # Return the unpacked argument structure of this signature,
    # discarding information about which arguments are semantically
    # related to each other.
    def arguments(self) -> Sequence[Binding]:
        return cpp.arguments(
            self.func.arguments, faithful=self.faithful,
            method=self.method, cpp_no_default_args=self.cpp_no_default_args)

    def name(self) -> str:
        n = cpp.name(self.func, faithful_name_for_out_overloads=self.faithful)
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # Render the C++ declaration for this signature
    def decl(self, *, name: Optional[str] = None, prefix: str = "", is_redispatching_fn: bool = False) -> str:
        returns_type = cpp.returns_type(self.func.returns).cpp_type()
        cpp_args = [a.decl() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(self, *, name: Optional[str] = None, prefix: str = "", is_redispatching_fn: bool = False) -> str:
        returns_type = cpp.returns_type(self.func.returns).cpp_type()
        cpp_args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    def ptr_type(self) -> str:
        args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{cpp.returns_type(self.func.returns).cpp_type()} (*)({args_types_str})'

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{cpp.returns_type(self.func.returns).cpp_type()} ({args_types_str})'


# Represents group of all CppSignatures associated with a
# FunctionSchema.  Right now, that's the regular, user-visible
# signature, as well as a "faithful" signature which doesn't
# have grouping.
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: Optional[CppSignature]

    def most_faithful_signature(self) -> CppSignature:
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    @staticmethod
    def from_native_function(f: NativeFunction, *, method: bool, fallback_binding: bool = False) -> 'CppSignatureGroup':
        func = f.func
        faithful_signature: Optional[CppSignature]
        if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
            faithful_signature = CppSignature(
                func=func,
                faithful=True,
                method=method,
                fallback_binding=fallback_binding,
                cpp_no_default_args=f.cpp_no_default_args
            )
        else:
            faithful_signature = None
        signature = CppSignature(
            func=func,
            faithful=False,
            method=method,
            fallback_binding=fallback_binding,
            cpp_no_default_args=f.cpp_no_default_args
        )
        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
        )

@dataclass(frozen=True)
class DispatcherSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Allows you to prepend an arbitrary prefix to the signature name.
    # This is useful for parts of the codegen that generate wrappers around kernels,
    # and need to avoid naming collisions.
    prefix: str = ""

    def arguments(self) -> List[Binding]:
        return dispatcher.arguments(self.func)

    def name(self) -> str:
        return self.prefix + dispatcher.name(self.func)

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None, *, is_redispatching_fn: bool = False) -> str:
        args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            args = ['c10::DispatchKeySet dispatchKeySet'] + args
        args_str = ', '.join(args)
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def exprs(self) -> List[Expr]:
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    def returns_type(self) -> CType:
        return dispatcher.returns_type(self.func.returns)

    def ptr_type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})'

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{self.returns_type().cpp_type()} ({dispatcher_args_types_str})'

    @staticmethod
    def from_schema(func: FunctionSchema, *, prefix: str = '') -> 'DispatcherSignature':
        return DispatcherSignature(func, prefix)

@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    prefix: str = ""

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns).cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns).cpp_type()} {name}({args_str})"

    def ptr_type(self) -> str:
        # don't include defaults in type signature!
        args_str = ', '.join(a.defn() for a in self.arguments())
        return f'{native.returns_type(self.func.returns).cpp_type()} (*)({args_str})'

    def arguments(self) -> List[Binding]:
        return native.arguments(self.func)

    def returns_type(self) -> CType:
        return native.returns_type(self.func.returns)

    def dispatcher_exprs(self) -> List[Expr]:
        return translate.translate(self.arguments(), dispatcher.arguments(self.func), method=False)


# Helper functions

def kernel_signature(
        f: NativeFunction, backend_index: BackendIndex, *, prefix: str = '') -> Union['NativeSignature', 'DispatcherSignature']:
    # Note [External Backends Follow Dispatcher API]
    # Kernel signatures for in-tree backends follow the "native" API,
    # while kernels for out-of-tree backends follow the dispatcher API.
    # See the comments in `native.py` for details, but historically there have been
    # some small differences in schema convention between them and the Dispatcher API.
    # Any differences that require translating between the two will results in a runtime cost,
    # so we'd like to keep the differences as small as possible.
    # With external backends, we'd like to enforce that they write their kernels with schemas
    # that match the Dispatcher API directly, if they can.
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, prefix=prefix)
    else:
        return NativeSignature(f.func, prefix)

# Functions only, no types
from tools.codegen.api import cpp, dispatcher, native, translate
