from math import *
from random import *

print("trouvez entre 1 et 20 \n")
j1 = int(input("j1, devine le nombre :"))
j2 = int(input("j2, devine le nombre :"))
j3 = int(input("j3, devine le nombre :"))

n = randint(1,20)
print(" le nombre a trouver c etait : ", n)

e1 = abs(n-j1)
e2 = abs(n-j2)
e3 = abs(n-j3)

if e1 < e2 and e1 < e3:
    print("1 gagne")
elif e2 < e1 and e2 < e3:
    print("2 gagne")
elif e3 < e1 and e3 < e2:
    print("3 gagne")
else:
    print("match nul")
