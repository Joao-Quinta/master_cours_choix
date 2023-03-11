import numpy as np
ret = [100,101,102,103,104,105,106,107,108,109,101,111,112]


ret_m = [((ret[i]-ret[i-1])/ret[i-1]) for i in range(1,len(ret))]
print(ret_m)
print(len(ret_m))

ret_a = np.prod([ret_m[i] + 1 for i in range(len(ret_m))]) - 1
print("anual return :",ret_a)

sum_monthly_ret = np.sum(ret_m)
print("sum monthly returns :",sum_monthly_ret)

avg_monthly_ret = np.mean(ret_m)
print("avearage of monthly returns :", avg_monthly_ret)

monthly_ret = np.power(1 + ret_a,1/12) - 1
print("avearage monthly ret (Rm)   :", monthly_ret)
