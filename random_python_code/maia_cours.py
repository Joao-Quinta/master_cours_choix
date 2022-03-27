def test1(liste):
    n_min = liste[0]
    for e in liste:
        if e < n_min:
            n_min = e
    print(n_min)


def test2(liste):
    n_min = liste[0]
    for e in liste:
        if e < n_min:
            n_min = e
    return n_min


def test3(liste):
    n_min = test2(liste)
    idx = []
    for i in range(len(liste)):
        if liste[i] == n_min:
            idx.append(i)
    return (n_min, idx)


# test1([4, 5, 3, 2, 6])
# va = test2([4, 5, 3, 2, 6])
# print(test3([4, 2, 3, 2, 6]))


############### exo 9
def accro():
    age = int(input("donnez votre age : "))
    taille = int(input("donnez votre taille en cm : "))
    if age < 3 or taille < 135:
        parcours = ["mini M1", "mini M2"]
        prix = "13 chf"
    elif age < 5:
        parcours = ["mini M1", "mini M2", "mini M3"]
        prix = "15 chf"
    elif age < 7 or taille < 150:
        parcours = ["P1", "P2", "P3", "P4"]
        prix = "26 chf"
    elif age < 18:
        parcours = ["P5", "P6", "P7", "P8", "P9"]
        prix = "29 chf"
    else:
        arcours = ["P5", "P6", "P7", "P8", "P9"]
        prix = "39 chf"
    return parcours, prix


# print(accro()[0])  # appeller la fonction

# palyndrome
def pal(chaine):
    res = True
    print("avant : ", chaine)
    chaine = chaine.replace(" ", "")# remplace le premier argument par le 2eme
    print("apres : ", chaine)
    chaine_renverse = chaine[::-1]
    print("apres renverse : ", chaine_renverse)
    return res


pal("salut j")
