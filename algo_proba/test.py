from random import *
fruits1 = ["pomme","orange","banane", "kiwi","fraise"]#1
print(fruits1[3])#2
fruits1.append("framboise")#3
print(fruits1)

fruits1.insert(0,"litchi")#4
print(fruits1)
fruits1.pop(2)#5
print(fruits1)

if("poire" in fruits1):
    print("oui")
else:
    print("non")#6

fruits2=["poire","ananas","pêche"]
print(fruits2)

fruits3 = fruits1 + fruits2#8
print(fruits3)





liste_aleatoire=[]

for i in range(8):
    #print(i)
    aleatoire = randint(-200,200)
    liste_aleatoire.append(aleatoire)
print(liste_aleatoire)

positifs=[]
negatifs=[]
for valeur in liste_aleatoire:

    if valeur<0:
        negatifs.append(valeur)

    else:
        positifs.append(valeur)

print(positifs)
print(negatifs)