#!/usr/bin/env python3

def display( name: str, val: int )->str:
    return name + "=" + val

def head( x: list[int] )->int:
    if len(x) > 0:
        return x[0]

x = {3}

y = head( x )

display( "y", y )
