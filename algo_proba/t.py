interet = 1.1
temps_investissement = 8
month = 2000
temps_retraite = 30

annee = 0
total = 0
argent_investi = 0
while annee < temps_retraite:
    if annee < temps_investissement:
        total = total + month * 12
        argent_investi = 2000 + argent_investi + month * 12

    total = total * interet
    annee = annee + 1

print("total ", total)
print("argent_investi ", argent_investi)



