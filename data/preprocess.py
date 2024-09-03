import pandas as pd
import numpy as np
import collections

df = pd.read_csv('predicted.csv', nrows=1000)
all_data = pd.read_csv('study_data_test_set.csv', nrows=1000)
data = pd.read_csv('./efficient_frontier.csv')
# print(all_data)

n = len(df)
df.insert(len(df.columns), "Website", all_data['Study URL'][:n])
df.insert(len(df.columns), "NCT", all_data['NCT Number'][:n])
df.insert(len(df.columns), "Condition", all_data['Conditions'][:n])
# df.insert(len(df.columns), "OtherInfo", ["cancer" for i in range(len(df))])

df.to_csv('predicted_out.csv', index=False)

risk_return_df = df.copy()
returns = data['Return']
risks = data['Risk']

n = len(data)
risk = np.array([(returns[i]**2 * (1-risks[i]) + risks[i]) - (returns[i] * (1-risks[i]) - risks[i])**2 for i in range(n)])
dummy_risk = []
dummy_return = []
dummy_color = []
num_pts = 1000

import math

ind_map = collections.OrderedDict()
ind_map = {
    data['Risk'][i] : int(i) for i in range(n)
}

for i in range(num_pts):
    rand_risk = .22 + (.99-.22)*np.random.random()
    ret = returns[ind_map[round(math.floor(rand_risk * 100) / 100, 2)]]
    noise = abs(np.random.normal(0, 0.15 * ret))
    dummy_risk.append(rand_risk)
    dummy_return.append(ret - noise)
    dummy_color.append(noise)

pd.DataFrame(zip(dummy_risk, dummy_return, dummy_color), columns=['Risk', 'Return','Color']).to_csv("scatter.csv", index=False)

