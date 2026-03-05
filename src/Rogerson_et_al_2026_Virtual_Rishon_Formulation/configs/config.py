from typing import Union, Sequence, TypeVar, Generic

T = TypeVar("T")

class Sequenceable(Generic[T]):
    def __class_getitem__(cls, item):
        return Union[item, Sequence[item]]
    