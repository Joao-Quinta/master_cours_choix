ch = "pluto"
print(ch)
t = list(ch)
print(t)
t[0] = "L"
print(t)
ch_nouveau = '|'.join(t)
print(ch_nouveau)

z = slice(0,6,2)
print(z)
print(list(ch_nouveau)[z])