from typing import Optional, Literal, Iterator, Union


def facGen(n: int) -> Iterator[int]:
    if n == 0 or n == 1:
        yield 1
    else:
        cur = 1
        for i in range(1, n + 1):
            cur = cur * i
            yield cur


def fizzBuzz(n) -> Iterator[Union[int, str]]:
    for i in range(1, n + 1):
        word = ""
        if i % 3 == 0:
            word = word + "Fizz"
        if i % 5 == 0:
            word = word + " Buzz"
        if word == "":
            yield i
        else:
            yield word
