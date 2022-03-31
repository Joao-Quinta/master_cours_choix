# exercice 1

def cacher_mot(o):
    mot_cache = ""  # mot vide
    for i in range(len(o)):
        mot_cache = mot_cache + "-"  # "--" + "-" -> "---"
    return mot_cache


# print(cacher_mot("in"))


# exercice 2

def est_nouvelle(lettre, lettre_utilise):
    #print(lettre)
    #print(lettre_utilise)
    if lettre in lettre_utilise:
        print("lettre deja utilise")
        return False, lettre_utilise
    else:
        lettre_utilise.append(lettre)
        return True, lettre_utilise


#print(est_nouvelle("c", ["a", "b"]))

# exercice 3
def est_bonne_lettre(lettre,mot):
    if lettre in mot:
        return True
    else:
        return False

#print(est_bonne_lettre("i","info"))

# exercice 4

def ajouter_lettre(lettre, mot, mot_cache):
    mot_cache = list(mot_cache)
    for i in range(len(mot)):
        if mot[i] == lettre:
            mot_cache[i] = lettre
    mot_cache = "".join(mot_cache)
    return mot_cache


mot = "info"
mot_cache = cacher_mot(mot)
print(ajouter_lettre("i", mot, mot_cache))