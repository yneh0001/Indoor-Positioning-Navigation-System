from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from abc import ABCMeta
from sklearn.svm import SVC, SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

build_acc = []
floor_acc = []

long_scaler = MinMaxScaler()
lat_scaler = MinMaxScaler()
x = 1
x2 = 1

def Pre_Process(dataframe):
    # # PRE-PROCESS
    # train_data_frame.drop_duplicates(subset=['ap', 'x', 'y', 'z'], inplace=True)
    # # test_data_frame.drop_duplicates(subset=['ap', 'x', 'y', 'z'], inplace=True)
    # train_data_frame = train_data_frame.dropna()

    A_list = []
    B_list = []
    C_list = []
    D_list = []

    for i in range(len(dataframe)):
        signal_index = 1
        var = dataframe.iloc[i, signal_index]

        if dataframe['ap'].iloc[i] == 'A':
            A_list.append(var)
            B_list.append(0)
            C_list.append(0)
            D_list.append(0)

        elif dataframe['ap'].iloc[i] == 'B':
            A_list.append(0)
            B_list.append(var)
            C_list.append(0)
            D_list.append(0)

        elif dataframe['ap'].iloc[i] == 'C':
            A_list.append(0)
            B_list.append(0)
            C_list.append(var)
            D_list.append(0)

        elif dataframe['ap'].iloc[i] == 'D':
            A_list.append(0)
            B_list.append(0)
            C_list.append(0)
            D_list.append(var)

    data = {'A': A_list,
            'B': B_list,
            'C': C_list,
            'D': D_list,
            'x': dataframe['x'],
            'y': dataframe['y'],
            'z': dataframe['z'],
            }

    new_df = pd.DataFrame(data)
    print(new_df)
    return new_df

def Load(path):
    train_data = pd.read_csv(path)
    test_data = pd.read_csv(path)

    # PRE-PROCESS
    train_data.drop_duplicates(subset=['ap', 'x', 'y', 'z'], inplace=True, keep='first')
    train_data = train_data.dropna()
    #
    test_data.drop_duplicates(subset=['ap', 'x', 'y', 'z'], inplace=True, keep='last')
    test_data = test_data.dropna()

    train_data_frame = Pre_Process(train_data)
    test_data_frame = Pre_Process(test_data)
    # test_data_frame = train_data_frame

    rest_data_frame = train_data_frame
    valid_data_trame = pd.DataFrame(columns=train_data_frame.columns)
    valid_num = int(len(train_data_frame) / 10)

    sample_row = rest_data_frame.sample(valid_num)
    rest_data_frame = rest_data_frame.drop(sample_row.index)

    valid_data_trame = pd.concat([valid_data_trame, sample_row])
    train_data_frame = rest_data_frame

    training_x = train_data_frame.to_numpy().T[:4].T
    training_y = train_data_frame.to_numpy().T[[2, 3, 4], :].T

    validation_x = valid_data_trame.to_numpy().T[:4].T
    validation_y = valid_data_trame.to_numpy().T[[2, 3, 4], :].T

    testing_x = test_data_frame.to_numpy().T[:4, 300:600].T
    testing_y = test_data_frame.to_numpy().T[[2, 3, 4], 300:600].T

    return training_x, training_y, validation_x, validation_y, testing_x, testing_y



def normalize_x(x_array):
    res = np.copy(x_array).astype(np.float)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if res[i][j] == 100:
                res[i][j] = 0
            else:
                res[i][j] = -0.01 * res[i][j]
    return res


def normalize_y(longs, lats):
    global long_scaler
    global lat_scaler
    longs = np.reshape(longs, [-1, 1])
    lats = np.reshape(lats, [-1, 1])
    long_scaler.fit(longs)
    lat_scaler.fit(lats)
    return np.reshape(long_scaler.transform(longs), [-1]), \
           np.reshape(lat_scaler.transform(lats), [-1])


def reverse_normalizeY(longs, lats):
    global long_scaler
    global lat_scaler
    longs = np.reshape(longs, [-1, 1])
    lats = np.reshape(lats, [-1, 1])
    return np.reshape(long_scaler.inverse_transform(longs), [-1]), \
           np.reshape(lat_scaler.inverse_transform(lats), [-1])


class Model(object):
    __metaclass__ = ABCMeta

    # ML model object
    longitude_regression_model = None
    latitude_regression_model = None
    floor_classifier = None

    # Training data
    normalize_x = None
    longitude_normalize_y = None
    latitude_normalize_y = None

    def __init__(self):
        pass

    def _preprocess(self, x, y):
        self.normalize_x = normalize_x(x)
        # remove nan
        self.normalize_x = np.nan_to_num(self.normalize_x)

        self.longitude_normalize_y, self.latitude_normalize_y = normalize_y(y[:, 0], y[:, 1])
        # remove nan
        self.longitude_normalize_y = np.nan_to_num(self.longitude_normalize_y)
        self.latitude_normalize_y = np.nan_to_num(self.latitude_normalize_y)

        self.floorID_y = y[:, 2]
        self.floorID_y = np.nan_to_num(self.floorID_y)
        # print(self.floorID_y)
        # self.buildingID_y = y[:, 3]

    def fit(self, x, y):
        # Data pre-processing
        self._preprocess(x, y)
        self.longitude_regression_model.fit(self.normalize_x, self.longitude_normalize_y)
        self.latitude_regression_model.fit(self.normalize_x, self.latitude_normalize_y)
        # print(np.isnan(self.normalize_x))
        # print(np.isnan(self.floorID_y))
        # print(np.isnan(self.floorID_y))
        # print(np.isnan(self.floorID_y))
        # print(self.floorID_y)
        # self.floor_classifier.fit(self.normalize_x, self.floorID_y)

    def predict(self, x):
        # Testing
        x = normalize_x(x)
        predict_longitude = self.longitude_regression_model.predict(x)
        predict_latitude = self.latitude_regression_model.predict(x)

        # predict_floor = self.floor_classifier.predict(x)

        # Reverse normalization
        predict_longitude, predict_latitude = reverse_normalizeY(predict_longitude, predict_latitude)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1),
                              np.expand_dims(predict_latitude, axis=-1)), axis=-1)
        # res = np.concatenate((res, np.expand_dims(predict_floor, axis=-1)), axis=-1)
        return res

    def error(self, x, y):
        _y = self.predict(x)
        dist = np.sqrt(np.square(_y[:, 0] - y[:, 0]) + np.square(_y[:, 1] - y[:, 1]))
        plot_dist_error(dist)
        map_plot(_y, y)
        print("Min: " + str(min(dist)))
        print("Mean: " + str(np.mean(dist)))
        print("Max: " + str(max(dist)))
        # print(min(dist), np.mean(dist), max(dist))

        return dist


class SVM(Model):
    def __init__(self):
        super().__init__()
        self.longitude_regression_model = SVR(verbose=True)
        self.latitude_regression_model = SVR(verbose=True)
        self.floor_classifier = SVC(verbose=True)

class RandomForest(Model):
    def __init__(self):
        super().__init__()
        self.longitude_regression_model = RandomForestRegressor()
        self.latitude_regression_model = RandomForestRegressor()
        self.floor_classifier = RandomForestClassifier()
        # self.building_classifier = RandomForestClassifier()


class GradientBoostingDecisionTree(Model):
    def __init__(self):
        super().__init__()
        self.longitude_regression_model = GradientBoostingRegressor()
        self.latitude_regression_model = GradientBoostingRegressor()
        self.floor_classifier = GradientBoostingClassifier()
        # self.building_classifier = GradientBoostingClassifier()

def plot_dist_error(dist):
    dist = dist.tolist()
    t = np.arange(0.0, len(dist), 1)
    y_mean = [np.mean(dist)] * len(dist)

    fig, ax = plt.subplots()
    ax.plot(t, dist, label='Error value')
    ax.plot(t, y_mean, label='Mean', linestyle='--', color='red')

    ax.set(xlabel='Data points (ID)', ylabel='Error (m)',
           title='Error value of points using baseline')
    ax.grid()

    global x
    fig.savefig("error" + str(x) + ".png")
    x += 1
    if(x > 3):
        x = 1
    plt.legend()
    plt.show()

def map_plot(_y, y):
    # take the first two features
    h = .02  # step size in the mesh

    # Calculate min, max and limits
    x_min, x_max = y[:, 0].min() - 1, y[:, 0].max() + 1
    y_min, y_max = y[:, 1].min() - 1, y[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Put the result into a color plot
    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], alpha=0.5, s=7, marker='o', label='actual position')
    plt.scatter(_y[:, 0], _y[:, 1], alpha=0.3, s=4, marker='o', label='predicted position')
    plt.xlim(xx.min() - 20, xx.max() + 20)
    plt.ylim(yy.min() - 20, yy.max() + 20)
    plt.legend()
    plt.grid()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("Data points")
    plt.tight_layout(pad=0)
    global x2
    plt.savefig('data_points' + str(x2) + '.png')
    x2 += 1
    if(x2 > 3):
        x2 = 1
    plt.show()


if __name__ == '__main__':
    path = 'rssi.csv'
    # Load(path)
    train_x, train_y, valid_x, valid_y, test_x, test_y = Load(path)

    # Training
    print("\n===========SVM===========")
    SVM = SVM()
    SVM.fit(train_x, train_y)
    SVM.error(test_x, test_y)

    print("\n===========RF===========")
    RF = RandomForest()
    RF.fit(train_x, train_y)
    RF.error(test_x, test_y)

    print("\n===========GBD===========")
    GBDT = GradientBoostingDecisionTree()
    GBDT.fit(train_x, train_y)
    GBDT.error(test_x, test_y)