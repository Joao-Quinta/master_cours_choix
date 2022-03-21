"""
x = [1, 2, 3, 4]
# index  0, 1, 2, 3
# index -4,-3,-2,-1
print(x[0], x[-4])

x = [1, 2, 3, 4, "True", True]
# index 0 = 1, index 1 = 2 ...
print(x[0])
print(x[1:])
print(x[0:1])
print(x[0:2])
print(x[1:3])
# 0:1 -> [0 ... 1[
# 0:2 -> [0 ... 2[
# 1:2 -> [1 ... 2[

# 2:4 -> 2,3
# 50:100 -> 50,51 ... 99

# 1:4 -> 1,2,3
# 1:3 -> 1,2


nom_de_liste = [1, 2, 3, 4, "True", True]
deuxieme_liste = [5, 6, 7, 8, 9]

long = len(nom_de_liste)
print("liste debut")
print(nom_de_liste)
print("liste apres apped 50")
nom_de_liste.append(50)  # aujoute élément à la fin de la liste
print(nom_de_liste)
print("liste apres insert 100 à 0")
nom_de_liste.insert(0, 100)  # ajoute 100 à la position 0
print(nom_de_liste)
print("liste apres pop")
nom_de_liste.pop()  # enleve dernier élément
print(nom_de_liste)
print("liste apres pop -2")
nom_de_liste.pop(-2)
print(nom_de_liste)





print(True or True and False)
# gauche à droite
# False
# gauche à droite, mais priorité à (and)
# True
have_umbrella = False
rain_level = 10
have_hood = True
is_workday = True
prepared_for_weather = have_umbrella or (rain_level < 5 and have_hood) or not (rain_level > 0 and is_workday)

# True
print(prepared_for_weather)

y = 5
# diviser y par x
x = 1
print("salut")  #

if x == 0:
    print("vaut 0, donc peut pas diviser")
elif x == 1:
    print("pas besoin de diviser car y/1 = y")
    print(y)
elif x == 3:
    print("x vaut 3")

else:
    print(y / x)
    print("vaut pas 0")

print("fini")  #

x = 1
if x > 0:
    print(x, "x plus grand que 0")
elif x >= 0:
    print(x, "x plus garnd egal 0")
else:
    print(x, "x plus petit que 0")

# tout chiffre est vrai, sauf 0
# toute chaine de charactère est vrai, sauf ""

# int(True) = 1, int(False) = 0
# 0, 1, 2, 3
#
ket = False
mus = False
oni = False


def fonction(ket, mus, oni):
    somme = int(ket) + int(mus) + int(oni)
    return somme == 1


print(fonction(False, False, False))  # tout false -> False
print(fonction(True, False, False))  # 1 sauce -> True
print(fonction(True, True, False))  # 2 sauce -> False
print(fonction(True, True, True))  # 3 sauce -> False



def moyenne_liste(x):
    somme = 0
    i = 0
    while i < len(x):
        somme = somme + x[i]
        i = i + 1
    somme = somme / len(x)
    return somme


## matrices
m = [[1],
     [4, 5, 6],
     [7, 8, [9, 10, 11]]]
# 7 -> m[2][0]
# 9 -> m [2][2][0]

m1_notes = [[4, 5, 6], [1, 2, 3], [7, 8, 9], [10, 12, 11, 13, 20, 40, 60, 11, 20, 30]]

moyenne = m1_notes[0][0] + m1_notes[0][1] + m1_notes[0][2]
moy = m1_notes[0]  # -> [4,5,6]

# len(m1_notes) = 2 -> i = 0,1
for i in range(0, len(m1_notes), 1):
    print(m1_notes[i], " <- element || i -> ", i)
    for j in range(0, len(m1_notes[i]), 1):
        print("valeur -> ", m1_notes[i][j], "  ||   i -> ", i, "  ||   j -> ", j)




i = 0
while i < qqchose:
    j = 0 
    while j < qqchose:
        j = j + 1
    i = i + 1
    
    
i = 0  # init
while i < 4:  # cond d arret
    print(i)
    i = i + 1  # incrementation

print("mtn for")

for i in range(0, 4, 1):
    print(i)

# range(4) -> [0,1,2,3]



def s(a):
    print("liste -> ", a)
    for i in range(0, len(a), 1):
        print("i -> ", i, " valeur correspondante -> ", a[i])
    return len(a)


a = [1, 2]
b = [1, 2, 3, 4]
c = [1, 2, 3, 1000]
s(b)
s(c)

b.append(5)

s(b)



def has(n):
    for abc in n:
        print("valeur regarde -> ", abc)
        if (abc % 7) == 0:
            return True
    return False


print(has([1, 21, 31]))


print("salut")
for i in range(10):
    print()
    print(i)
    print("sonya")
    print()
print("a")


dico = {"prenom": "joao", "nom": "quinta", "age": 24}

print(dico.keys())

l = [1, 2, 3]
for valeur in l:
    print(valeur)

for cle, valeur in dico.items():
    print(cle, "     ", valeur)

a = "12345"
print(a.isdigit())


s = "There is a Casino."
mot = "CAsInO"

print("DEBUT : ")
print(s)
print(mot)
print()

s = s.lower()
mot = mot.lower()

print("post lower case : ")
print(s)
print(mot)
print()

s = s.replace(".", "")

print("post point final removal : ")
print(s)
print(mot)
print()

s = s.split()

print("post split : ")
print(s)
print(mot)
print()

if mot in s:
    print("oui mot est present")

else:
    print("pas presetn")

















liste_res = []

liste = ["phrase0", "phrase1", "phrase2"]
for i in range(len(liste)):
    print( i, liste[i])

    liste_res.append(i)



def addi(a, b):  # b = "casino" -> ["casino", "they"]
    return a + b


print(addi(1, "2"))

# -> [1,2], [1,2] -> [1+3, 2+4]


doc = ["salut casino.", "casino, they", "casinothey"]
key = ["casino", "they"]


def f(d, k):
    dico = {}
    for ke in k:
        l = []
        for i in range(len(d)):
            p = d[i]
            p = p.lower()
            p = p.replace(".", "").replace(",", "")
            p = p.split()
            if ke in p:
                l.append(i)
        dico[ke] = l
    return dico


def t(d, k):
    return {ke: [i for i in range(len(d)) if ke in d[i].lower().replace(".", "").replace(",", "").split()] for ke in k}


print(t(doc, key))

"""

import numpy

y = [[1, 2, 3], [4, 5, 6]]
print(y)
z = numpy.asarray(y)
print(z)

print(y[1][2])
print(z[1, 2])
