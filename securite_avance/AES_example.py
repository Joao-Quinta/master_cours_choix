from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad


# verified with --> https://www.javainuse.com/aesgenerator

def encryptMessageKey(message_, key_):
    return AES.new(key_, AES.MODE_ECB).encrypt(pad(message_, AES.block_size))


data = b'a'
print("data -> ", data.hex())
key = get_random_bytes(32)  # random 256 bits key
key = b"You can't see meYou can't see me"
print("key -> ", key.hex())
cipher = AES.new(key, AES.MODE_ECB)
ciphered_data = cipher.encrypt(pad(data, AES.block_size))
print("cipher len -> ", len(ciphered_data.hex()))
print("cipher -> ", ciphered_data.hex())

# decryption
cipher = AES.new(key, AES.MODE_ECB)
decrypted_cipher = cipher.decrypt(ciphered_data)
print(decrypted_cipher)
