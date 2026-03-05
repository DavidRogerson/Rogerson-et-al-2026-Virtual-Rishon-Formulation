from typing import Union, Type, Optional, Sequence, TypeVar, TypeAlias
from numpy import ndarray



#This is a type used if the config parameter makes sense to be sequenced in a tenpy sequentioal simulation
#This includes everything that is not breaking the mps structure etc.
SequenceableInt: TypeAlias = Union[int, Sequence[int]]
SequenceableFloat: TypeAlias = Union[float, Sequence[float]]
SequenceableStr: TypeAlias = Union[str, Sequence[str]]
SequenceableComplex: TypeAlias = Union[complex, Sequence[complex]]
SequenceableBool: TypeAlias = Union[bool, Sequence[bool]]
SequencableNdarray: TypeAlias = Union[ndarray, Sequence[ndarray]]

SequenceableBoolNone: TypeAlias = Union[Union[bool, None], Sequence[Union[bool, None]]]
SequenceableIntNone: TypeAlias = Union[Union[int, None], Sequence[Union[int, None]]]
SequenceableFloatNone: TypeAlias = Union[Union[float, None], Sequence[Union[float, None]]]
SequenceableStrNone: TypeAlias = Union[Union[str, None], Sequence[Union[str, None]]]
SequenceableComplexNone: TypeAlias = Union[Union[complex, None], Sequence[Union[complex, None]]]
SequenceableNdarrayNone: TypeAlias = Union[Union[ndarray, None], Sequence[Union[ndarray, None]]]