amount = 2000000
print("need -> ", amount * 0.2)
u_owe_ = 0.8 * amount
u_owe = 0.8 * amount
rate = 0.01
time = 30
pay_per_year = u_owe / time
print("amount owed -> ", u_owe)
print("time to pay (years) -> ", time)
print("rate -> ", rate * 100, "%")
print("amount paid per year (pre interest) -> ", pay_per_year)

paid_amount = 0
for i in range(time):
    pay_this_year = pay_per_year + (u_owe * rate)
    print("year -> ", i + 1, " you pay ", pay_this_year, " or ", int(pay_per_year / 12), " per month")
    paid_amount = paid_amount + pay_this_year
    u_owe = u_owe - pay_per_year

print("full paid amount -> ", paid_amount, "(", amount * 0.2 + paid_amount, ")")
print("amount bank won -> ", paid_amount - u_owe_, " per year -> ", (paid_amount - u_owe_) / time)
