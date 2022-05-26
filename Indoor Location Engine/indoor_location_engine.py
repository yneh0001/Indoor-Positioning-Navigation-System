import sys

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from abc import ABCMeta
from sklearn.svm import SVC, SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = 1
x2 = 1


def Pre_Process(dataframe):
    # # PRE-PROCESS
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
    # print(new_df)
    return new_df


def Load(path):
    train_data = pd.read_csv(path)
    test_data = pd.read_csv(path)

    # PRE-PROCESS
    # drop duplications and empty
    train_data.drop_duplicates(subset=['ap', 'x', 'y', 'z'], inplace=True, keep='first')
    train_data = train_data.dropna()
    # drop duplications and empty
    test_data.drop_duplicates(subset=['ap', 'x', 'y', 'z'], inplace=True, keep='last')
    test_data = test_data.dropna()

    train_data_frame = Pre_Process(train_data)
    test_data_frame = Pre_Process(test_data)
    rest_data_frame = train_data_frame
    valid_num = int(len(train_data_frame) / 10)

    sample_row = rest_data_frame.sample(valid_num)
    rest_data_frame = rest_data_frame.drop(sample_row.index)

    train_data_frame = rest_data_frame

    training_x = train_data_frame.to_numpy().T[:4].T
    training_y = train_data_frame.to_numpy().T[[4, 5, 6], :].T

    testing_x = test_data_frame.to_numpy().T[:4].T
    testing_y = test_data_frame.to_numpy().T[[4, 5, 6], :].T

    return training_x, training_y, testing_x, testing_y


class Model(object):
    __metaclass__ = ABCMeta

    # ML model object
    longitude_regression_model = None
    latitude_regression_model = None
    floor_classifier = None

    # Training data
    x = None
    longitude_y = None
    latitude_y = None

    def __init__(self):
        pass

    def _preprocess(self, x, y):
        self.x = x
        # remove nan
        self.x = np.nan_to_num(self.x)

        self.longitude_y, self.latitude_y = y[:, 0], y[:, 1]
        # remove nan
        self.longitude_y = np.nan_to_num(self.longitude_y)
        self.latitude_y = np.nan_to_num(self.latitude_y)

        self.floorID_y = y[:, 2]
        self.floorID_y = np.nan_to_num(self.floorID_y)

    def fit(self, x, y):
        # Data pre-processing
        self._preprocess(x, y)
        self.longitude_regression_model.fit(self.x, self.longitude_y)
        self.latitude_regression_model.fit(self.x, self.latitude_y)

    def predict(self, x):
        # Testing
        predict_longitude = self.longitude_regression_model.predict(x)
        predict_latitude = self.latitude_regression_model.predict(x)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1),
                              np.expand_dims(predict_latitude, axis=-1)), axis=-1)
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


class GradientBoostingDecisionTree(Model):
    def __init__(self):
        super().__init__()
        self.longitude_regression_model = GradientBoostingRegressor()
        self.latitude_regression_model = GradientBoostingRegressor()
        self.floor_classifier = GradientBoostingClassifier()


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
    if (x > 3):
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
    if (x2 > 3):
        x2 = 1
    plt.show()


def managing_dataset_menu():
    print("\n===========Main Menu===========")
    print("1.Managing Dataset")
    print("2.Quit")
    print("\nPlease enter your choice: ")
    choice = input()
    if int(choice) == 1:
        try:
            print("\nPlease enter your dataset name with .csv extension: ")
            path = input()
            print("Loading dataset...")
            # path = 'rssi.csv'
            # Load(path)
            train_x, train_y, test_x, test_y = Load(path)
            print("Dataset loaded successfully!")

            # call training
            training_menu(train_x, train_y, test_x, test_y)

        except:
            print("Failure! Please check you have entered the correct input.")

    else:
        print("\nThank you!")

def training_menu(train_x, train_y, test_x, test_y):
    print("\n===========Training Menu===========")
    print("1.Train Support Vector Machine")
    print("2.Train Random Forest")
    print("3.Train Gradient Boosting Decision Tree")
    print("4.Quit")
    print("\nPlease enter your choice: ")
    choice2 = input()
    if int(choice2) == 1:
        # Training
        SVM.fit(train_x, train_y)
        SVM.error(test_x, test_y)
        training_menu(train_x, train_y, test_x, test_y)

    elif int(choice2) == 2:
        RF.fit(train_x, train_y)
        RF.error(test_x, test_y)
        training_menu(train_x, train_y, test_x, test_y)

    elif int(choice2) == 3:
        GBDT.fit(train_x, train_y)
        GBDT.error(test_x, test_y)
        training_menu(train_x, train_y, test_x, test_y)
    else:
        print("\nThank you!")


if __name__ == '__main__':
    SVM = SVM()
    RF = RandomForest()
    GBDT = GradientBoostingDecisionTree()

    managing_dataset_menu();
