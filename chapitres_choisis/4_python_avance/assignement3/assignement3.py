from typing import Optional, Literal, Iterator, Union
from dataclasses import dataclass


def facGen(n: int) -> Iterator[int]:
    yield 1
    if n == 1:
        yield 1
    else:
        cur = 1
        for i in range(1, n + 1):
            cur = cur * i
            yield cur


def fizzBuzz(n: int) -> Iterator[Union[int, str]]:
    for i in range(1, n + 1):
        word = ""
        if i % 3 == 0:
            word = word + "Fizz"
        if i % 5 == 0:
            word = word + "Buzz"
        if word == "":
            yield i
        else:
            yield word


def movingAvg(size: int) -> Iterator[float]:
    avg = [i for i in range(10)]
    for i in range(len(avg) - size + 1):
        s = 0
        for j in range(size):
            s = s + avg[i + j]
        yield s / size


def primeNumbers(n: int) -> Iterator[int]:
    allV = [str(i) for i in range(2, n)]
    for i in range(len(allV)):
        if allV[i][-1] != "-":
            v = int(allV[i])
            val = int(allV[i])
            yield v
            while v < len(allV) + 2:
                allV[v - 2] = allV[v - 2] + "-"
                v = v + val


@dataclass
class Tree:
    value: int
    left: Optional['Tree']
    right: Optional['Tree']


def dfs(t: Tree) -> Iterator[int]:
    yield t.value
    if t.left is not None:
        dfs(t.left)
    if t.right is not None:
        dfs(t.right)


def bfs(t: Tree) -> Iterator[int]:
    next = [t]
    while len(next) > 0:
        t1 = next.pop(0)
        yield t1.value
        if t1.left is not None:
            next.append(t1.left)
        if t1.right is not None:
            next.append(t1.right)
