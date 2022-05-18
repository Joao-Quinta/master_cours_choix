from typing import Optional, Literal, Iterator, Dict, Union, cast, Callable, Any, TypeVar
from dataclasses import dataclass

globalCounter: Dict[str, int] = {}
globalComputations: Dict[str, Any] = {}

F = TypeVar('F', bound=Callable[..., Any])


def count(f: F) -> F:
    name = f.__name__

    def g(x) -> None:
        keys = globalCounter.keys()
        if f.__name__ in keys:
            globalCounter[f.__name__] = globalCounter[f.__name__] + 1
        else:
            globalCounter[f.__name__] = 1
        return f(x)

    return cast(F, g)


def cached(f):
    res = {}
    def g(x):
        if x not in res:
            print()

@count
def t(i: int) -> int:
    return i + 1


t(1)
print(globalCounter)
