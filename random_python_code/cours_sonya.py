def carre(x):  # 5 -> 25
    print("carre a ete appele")
    return {x: x * x}


def carre2(x):  # 5 -> 26
    print("carre2 a ete appele")
    return x * x + 1


y, y2 = carre(5), carre2(5)
print(y)
print(y2)

e = [1, 1, 2, 3, 4, 5]
print(e)
e1 = set(e)
print(e1)


def ca(x, y, z, t, a, s, e, q, w, u, i):
    return e, z


l = ca(1, 1, 5, 1, 1, 1, 4, 1, 1, 1, 1)
print(l)
print(l[0])
print(l[1])





z = s()
print(z)





c1 = (e['GENDER']=='FEMALE' & e['BASE_SALARY'] > 100000)
c1 = (e['GENDER']=='MALE' & e['BASE_SALARY'] < 50000)
c_all = c1 | c2
e_8 = e[e['GENDER']=='FEMALE' & e['BASE_SALARY'] > 100000] + e[e['GENDER']=='MALE' & e['BASE_SALARY'] < 50000]



employee[c1 & c2].head()

ctout = c1 | c2 | c3
e[ctout]

x = "1"
x = int(x)