
from random import *
"""
from turtle import *


def carre(taille, largeur):
    down()
    width(largeur)
    for i in range(4):
        forward(taille)
        left(90)
    up()



carre(5,3)
"""


def somme(a, b):
    s = a + b
    return s


res = somme(2, 3)



def guideTaille(nombre):
    if nombre < 38:
        return "S"
    if 38 < nombre and nombre > 42:
        return "M"
    if nombre > 42:
        return "L"


#valeur = int(input("donne taille en chiffre"))
#print(valeur, "correspond à du", guideTaille(valeur))


def predicition():
    liste_0=[]
    liste_1=[]
    for i in range(11):
        y=randint(0,1)
        if y==0:
            liste_0.append(y)
        else:
            liste_1.append(y)
        if len(liste_0) > len(liste_1) :
            return True
        else:
            return False
print(predicition())

def pred():
    y = randint(0,1)
    return y

def realPred():
    liste_0 = []
    liste_1 = []
    for i in range(11):
        v = pred()
        if v==0:
            liste_0.append(v)
        else:
            liste_1.append(v)
    if len(liste_0)>len(liste_1):
        return True
    else:
        return False