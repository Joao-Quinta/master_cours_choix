etatPair = []
etatImpair = []


def list_pair(b):
    l_pair = []
    for i in range(len(b)):  # parcourt les rangées
        for j in range(b[i]):  # parcourt le nb allu dans chaque rangées
            pair = [i + 1, j + 1]  # on veut au moins 1 allu et on a 4 rangées (de 1,2,3,4)
            l_pair.append(pair)  # on met la pair dans notre liste contenant toutes les pairs
    return l_pair


def sum_col(b):
    m = max(b)
    list_bin = []
    size = len(format(m, 'b'))  # calcule la taille de l'écriture binaire de notre maximum
    for i in range(len(b)):  # création de notre liste de binaire
        bin = format(b[i], '0' + str(
            size) + 'b')  # convertit les nombres en binaires et uniformise leur longueur avec celle du maximum
        list_bin.append(bin)
    sum = []
    for j in range(size):  # parcourt les digit de notre binaire
        n = 0
        for e in range(len(b)):  # parcourt les elements binaires dans list_bin et additionne digit par digit
            n = n + int((list_bin[e])[j])  # faire la somme de chaque colonne
        sum.append(n)  # liste qui contient la somme pour chaque colonne
    return sum


def new_boite(b, pair):
    b_copy = b.copy()  # on fait une copie car on ne veut pas modifier notre "vrai" boite en enlevant des allu si ce n'est pas une bonne configuration
    rangee = pair[0]
    allu = pair[1]
    b_copy[rangee - 1] = b_copy[rangee - 1] - allu
    return b_copy


def config_gagnante(b):  # [2,2,4]
    if len(b) >= 2:  # tant que la boite est au moins 2
        for i in range(len(b)):  # parcourt la somme de chaque colonne de notre liste binaire
            if b[i] % 2 != 0:  # regarde si les elements de la boite sont impairs
                return False
        return True
    else:  # il reste 1 seul element dans la boite binaire
        if b[0] == 1 or b[0] == 3:  # il faut que la dernière colonne (2⁰) a comme valeur 1 ou 3
            return True
        else:
            return False


def makeEtats():
    stack = []
    seen = []
    f = [1, 3, 5, 7]
    seen.append(f)
    stack.append(f)
    etatPair.append(f)
    while len(stack) > 0:
        c = stack.pop(0)
        seen.append(c)
        if config_gagnante(c):
            etatPair.append(c)
        else:
            etatImpair.append(c)
        t = list_pair(c)
        for tr in t:
            if tr not in seen:
                stack.append(new_boite(c, tr))


makeEtats()
print(etatPair)
print(etatImpair)
