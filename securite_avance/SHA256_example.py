from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad
from Cryptodome.Hash import SHA256

import AES_example

print()

h = SHA256.new()
h.update(b'Hello')
print(h.digest().hex())

h1 = SHA256.new()
h1.update(h.digest())
print(h1.digest().hex())

cipher = AES_example.encryptMessageKey(b'salut', h1.digest())
print(len(cipher.hex()), "   ->   ", cipher.hex())
