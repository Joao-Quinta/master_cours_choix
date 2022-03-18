def extended_euclid_gcd(a, b):
    if a > b:
        """
        Returns a list `result` of size 3 where:
        Referring to the equation ax + by = gcd(a, b)
            result[0] is gcd(a, b) -> d
            result[1] is x -> u
            result[2] is y -> v
        """
        s = 0
        old_s = 1  # u
        t = 1
        old_t = 0  # v
        r = b
        old_r = a

        while r != 0:
            a = old_r
            # print("-> ", old_r)
            quotient = old_r // r  # In Python, // operator performs integer or floored division
            # print("oldr = r * quotient -> ", old_r, r, quotient)
            # This is a pythonic way to swap numbers
            # See the same part in C++ implementation below to know more
            old_r, r = r, old_r - quotient * r  # c'est bon

            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t

            # print(quotient, " = ", old_r, r, " u -> {", old_s, s, "}", " v -> {", old_t, t, "}")
            print(r, " = ", a, " * ", old_s, " + ", old_r, " * ", old_t, " -> ", a * old_s + old_r * old_t)
        return [old_r, old_s, old_t]
    else:
        return extended_euclid_gcd(b, a)


print(extended_euclid_gcd(164, 72))
