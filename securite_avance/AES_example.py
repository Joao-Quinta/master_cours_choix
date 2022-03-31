from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad

# verified with --> https://www.javainuse.com/aesgenerator

data = b'Can you smell what the Rock is cooking?'
print("data -> ", data)
key = get_random_bytes(32)  # random 256 bits key
key = b"You can't see meYou can't see me"
print("key -> ", key)
cipher = AES.new(key, AES.MODE_ECB)
ciphered_data = cipher.encrypt(pad(data, AES.block_size))
print("cipher -> ", ciphered_data.hex())
