import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data(train, test, target_column):

    df_train = pd.read_excel(train)
    df_test = pd.read_excel(test)
    
    x_train = df_train.drop(columns=[target_column]).values # [100,4] delete final score from matrix
    y_train = df_train[target_column].values

    x_test = df_test.drop(columns=[target_column]).values # [100,4] delete final score from matrix
    y_test = df_test[target_column].values

    names = df_train.drop(columns=[target_column]).columns.tolist()
    
    return x_train, y_train, x_test, y_test, names

class LinearRegression:

    def __init__(self):
        self._params = None
        self.coeff = None
        self.feature_names = None
        self.x_train = None
        self.y_train = None
        
    def _gen_x_mat(self, x):
        one = np.ones((x.shape[0], 1))
        mat_x  = np.concatenate([x, one], axis = 1)
        return mat_x

    def _gen_y_mat(self, y):
        mat_y = y.reshape(-1,1)
        return mat_y
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y.reshape(-1, 1)
        mat_x = self._gen_x_mat(x)
        mat_y = self._gen_y_mat(y)
        self.coeff = self._cal_coef(mat_x, mat_y)

    def _cal_coef(self, mat_x, mat_y):
        first = np.linalg.inv(mat_x.T @ mat_x) # (x_T * x)^-1
        second = mat_x.T @ mat_y #(x_T * y)
        coeff = first @ second # (x_T * x)^-1  *  (x_T * y)
        self.coeff = coeff
        return coeff

    
    def predict(self, x):
        mat_x = self._gen_x_mat(x)
        predict = (mat_x @ self.coeff)
        return predict

    def cal_R(self, x, y):
        y_pred = self.predict(x)
        y_true = y.reshape(-1,1)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        R2 = 1 - (ss_res / ss_tot)
        return R2

    def weight_rank(self, feature_names):
        weights = self.coeff[:-1].reshape(-1) ## remove last coeff (constant)
        rank = []
        for i in range(len(weights)):
            rank.append([i, abs(weights[i])])  ## add index to each values weights
        
        rank.sort(key=lambda x: x[1], reverse = True)  ##sorting the weight using index 1 ( weight value ) at desceding ordere

        result = []
        for i in range(2):
            index = rank[i][0]    ## get the index num
            name = feature_names[index]   ## get the name of the corresponding index
            result.append((index, name, weights[index]))   ## add the names of the weights to list result
        return result

    def plot(self, x, y, feature_names):
        top_features = self.weight_rank(feature_names)

        plt.figure(figsize=(5, 4))

        for i in range(len(top_features)): ## loop 2 times for the top 2 weight
            index = top_features[i][0]  ## the index
            name = top_features[i][1]  ## the name
            x_plot = x[:, index]       ## this is the value of all X at the wight only ( ex: if we have midterm index, we only get X of midterm)
            y_plot = self.coeff[index][0] * x_plot + self.coeff[-1][0]   ## weight of index * x of index + constant  ( normal y = mx +c)

            plt.subplot(1, 2, i + 1)
            plt.plot(x_plot, y, 'bo')
            plt.plot(x_plot, y_plot, '-r')
            plt.xlabel(name, )
            plt.ylabel('final')
            plt.title(name + ' vs final')

        plt.show()



# if __name__ == '__main__':
    
#     train = "TrainMath.xlsx"
#     test = "TestMath.xlsx"
#     target_column = "final"

#     gg = LinearRegression()
#     x, y, x1, y1, names = data(train,test, target_column)
#     model = gg.fit(x, y)
#     test_pred = gg.predict(x1)

#     print("\n"+ target_column + " Score Predict")
#     print(test_pred)

#     print("\n""Coeff")
#     print(gg.coeff)
        
#     R2 = gg.cal_R(x,y)
#     print("\n""R^2")
#     print(R2)

#     weight = gg.weight_rank(names)
#     gg.plot(x, y, names)

    