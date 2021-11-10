from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


# csvファイルは任意のものを選ぶ
df = pd.read_csv("csv/Book5.csv")

x = df[["ggap"]]
y = df[["tfp"]]

model_lr = LinearRegression()
model_lr.fit(x,y)

plt.plot(x, y, 'o')
plt.plot(x, model_lr.predict(x), linestyle="solid")
# タイトルも任意のものを選ぶ
plt.title("Regression Analysis in ???")
plt.show()

print('モデル関数の回帰変数 w1: %.3f' %model_lr.coef_)
print('モデル関数の切片 w2: %.3f' %model_lr.intercept_)
print('y= %.3fx + %.3f' % (model_lr.coef_ , model_lr.intercept_))
print('決定係数 R^2： ', model_lr.score(x, y))