import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# create dataframe from csvfile.
cust_df = pd.read_csv("ggap_and_tfp.csv")

country_arr = np.array([cust_df['country'].tolist()]).flatten()

# delete country column.
del(cust_df['country'])


cust_arr = np.array([cust_df['ggap1980'].tolist(),
                    cust_df['ggap1985'].tolist(),
                    cust_df['ggap1990'].tolist(),
                    cust_df['ggap1995'].tolist(),
                    cust_df['ggap2000'].tolist(),
                    cust_df['ggap2005'].tolist(),
                    cust_df['ggap2010'].tolist(),
                    cust_df['ggap2015'].tolist(),
                    cust_df['range7580'].tolist(),
                    cust_df['range8085'].tolist(),
                    cust_df['range8590'].tolist(),
                    cust_df['range9095'].tolist(),
                    cust_df['range9500'].tolist(),
                    cust_df['range0005'].tolist(),
                    cust_df['range0510'].tolist(),
                    cust_df['range1015'].tolist()
                    ], np.float64)

cust_arr = cust_arr.T

pred = KMeans(n_clusters=4).fit_predict(cust_arr).flatten()

dic = {}

for idx, ele in zip(country_arr, pred):
    dic[idx] = ele

group_A = []
group_B = []
group_C = []
group_D = []

for country, val in dic.items():
    if val == 0:
        group_A.append(country)
    elif val == 1:
        group_B.append(country)
    elif val == 2:
        group_C.append(country)
    else:
        group_D.append(country)

print("Group_A:", group_A)
print("Group_B:", group_B)
print("Group_C:", group_C)
print("Group_D:", group_D)