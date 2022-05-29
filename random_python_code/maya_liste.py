from random import randint


def somme(maia):
    s = 0
    for i in range(len(maia)):
        s = s + maia[i]
    return s


def bingo(x):
    l = []
    for i in range(x):
        l.append(i + 1)
    print("l avant ->", l)
    while len(l) > 0:
        v = l.pop(randint(0, len(l) - 1))
        print(v)
    print("l apres -> ", l)


def compare(a, b):
    if a > b:
        return a
    else:
        return b


def trouve_max(l):
    max_actu = l[0]
    for i in range(len(l)):
        max_actu = compare(max_actu, l[i])
    return max_actu


l = [40, 50, 70, 2, 90, 80]
print(trouve_max(l))
