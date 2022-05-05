from collections import Counter


def ceaser(text, v1):
    t = ""
    for i in text:
        v = chr((ord(i) + v1) % 256)
        t = t + v
    return t


def addKey(text, key="Based"):
    t = ""
    for i in range(len(text)):
        j = i % len(key)
        v = (ord(text[i]) + ord(key[j])) % 256
        t = t + chr(v)
    return t


def addKey2(text, key="Based"):
    t = ""
    for i in range(len(text)):
        j = i % len(key)
        v = ((ord(text[i]) + ord(key[j]) - 97) % 26)+97
        t = t + chr(v)
    return t


def lookForThe(text):
    text = text.lower()
    for i in range(len(text) - 2):
        window = text[i:i + 3]
        for j in range(256):
            c = ""
            for z in window:
                c = c + chr((ord(z) + j) % 256)
            if c == "the":
                print(c)
                print(j)
                print(window)
                return c, j, window


text = "R3JhbmRwYSBzZW50IG1lIG9uIGEgdHJlYXN1cmUgaHVudCB0byBnZXQgYSBsb3N0IGZsYWcgYnV0IHRoZSBtYXAgd2FzbnQgY29tcGxldGVkLiBBbGwgSSBzYXcgd2FzIHRoZSBjaXR5IG9mIFJPVCBhbmQgYSBjb2RlZCBtZXNzYWdlOiBKdW5nIGxiaCBqdmZ1IHNiZSB2ZiBuZyAuLi4tIC4uIC0tLiAuIC0uIC4gLi0uIC4gLi4uIC0uLS4gLi0gLi4uIC0gLi0uLiAuIGhmciBndXIgeHJsIFFCZ2piIG5hcSBjZWJpdnFyIGd1ciBjdWVuZnIgVFBJQ0d7TXBjSHBrc2tLYmlman0="
text = text.lower()
print(len(text))
print(Counter(text))
print(ord("i"))
print(ord("e"))
print(addKey2(text, key="vigenere"))
