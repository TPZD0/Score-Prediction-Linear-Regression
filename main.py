from ScorePredict import *

train = "TrainMath.xlsx"
test = "TestMath.xlsx"
target_column = "final"

gg = LinearRegression()
x, y, x1, y1, names = data(train,test, target_column)
model = gg.fit(x, y)
test_pred = gg.predict(x1)

print("\n"+ target_column + " Score Predict")
print(test_pred)

print("\n""Coeff")
print(gg.coeff)
    
R2 = gg.cal_R(x,y)
print("\n""R^2")
print(R2)

weight = gg.weight_rank(names)
gg.plot(x, y, names)