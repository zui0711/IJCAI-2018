import pandas as pd

d1 = pd.read_csv('ans/LR 2018-04-19 09-51-25.csv', sep=' ')
d2 = pd.read_csv('ans/XGB 2018-04-19 22-38-29.csv', sep=' ')

dave = d1['predicted_score'] * 0.7 + d2['predicted_score'] * 0.3
d1['predicted_score'] = dave

d1.to_csv("ans/ans.csv", columns=["instance_id", "predicted_score"],
          index=False,
          sep=" ")