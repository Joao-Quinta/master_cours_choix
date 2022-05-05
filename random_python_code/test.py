# Creation du jeu


boite=[[1], [1,1,1], [1,1,1,1,1], [1,1,1,1,1,1,1]]

# Calcul de la taille de chaque rangée et de la boite

s_A=len(boite[0])
s_B=len(boite[1])
s_C=len(boite[2])
s_D=len(boite[3])
s_b=s_A + s_B + s_C + s_D # taille de la boite


# Demander si sont 2 joueurs ou 1 joueurs (plus tard)
# Le joueur choisit la rangée dans laquelle il veut enlever ses allumettes


# Faut changer pour qu'on puisse plus prendre dans la rangée si on a pas d'allumettes, cad si y a que des 0 dans la rangée
# y a un problème avec le nombre d allu qui reste, surement voir copy liste
nb_allu=0 # initalisation du nb d'allu contenu dans les rangées
while s_b !=1 : # tant qu'il reste plus d'allumette

    #print("ici", nb_allu)
    x=int(input("Choisis une rangée entre 0 et 3 : "))
    while (x!=0 and x!=1 and x!=2 and x!=3) :
      x=int(input("Choisis une rangée entre 0 et 3 : "))
    else :
        nb_allu=boite[x].count(1) # compte le nombre d'allumette dans la rangée
        print("nb_allu dans la rangée",x, "est",nb_allu)
        while nb_allu==0 :
            print("cette rangée est vide")
            x=int(input("Choisis une autre rangée : "))
            nb_allu=boite[x].count(1) # compte le nombre d'allu dans la nouvelle rangée Choisis
        else : # si la rangée n'est pas vide
            print("il y a", nb_allu, " allumettes dans la rangée", x) # compte le nombre d'allumettes qui reste dans la rangée choisis


    # Le joueur choisi le nombre d'allumettes qu'il veut enlever dans cette rangée


    y=int(input("Choisis le nombre d'allumettes a enlevé dans cette rangée : "))

    while y>nb_allu or y<0 :
      print("tu peux enlever au maximum",nb_allu, " allumettes")
      y=int(input("Choisis le nombre d'allumettes a enlevé dans cette rangée :"))
    if y==1 :
        boite[x][0]=0
        boite[x].sort(reverse=True)
        print(boite[x])
    else :
        for i in range(y) :
            print("i",i)
            boite[x][i]=0  # les allumettes enlevés correspondent à la valeur 0 dans la rangée
            print(boite[x])
            print("")
    boite[x].sort(reverse=True)
    nb_allu=nb_allu - y  # on a enlevé y allumettes à la rangée x
    print("il reste", nb_allu, "allu dans la rangée",x)
    s_b= s_b - y  # on a enlevé y allumettes à notre boite de 16 allumettes
    print("il reste", s_b, "allumettes au total")
    if s_b==0 :
        print("TU AS PERDU")
        exit()
else :
    print("TU AS PERDU")
    exit()