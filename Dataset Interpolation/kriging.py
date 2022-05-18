import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random

ot.Log.Show(ot.Log.NONE)

"""
read data from rssi.csv file and filter 
it to access point A in floor 1
"""

data = pd.read_csv('rssi.csv')
flt = data['ap'] == 'A'
data2 = data[flt]
data3 = data2['z'] == 1.0
data3 = data2[data3]

x = data3['x']
y = data3['y']
z = data3['signal']

# convert stream data to list of values
list_x = x.tolist()
list_y = y.tolist()
list_z = z.tolist()


# a list of lists of x y coordinates
x_y = [list(x) for x in zip(list_x, list_y)]

"""
filtering thousands of data points by
taking the average of redundant data
"""


def ftr(xy):
    lst = []
    count = 0
    average = 0

    # loop and take the average if redundant
    for i in range(0, len(x) - 1):
        average = list_z[i]
        count = 1
        if x_y[i] == x_y[i + 1]:
            count += 1
            average += list_z[i + 1]
            i += 1
            continue
        else:
            lst.append([x_y[i - 1], average / count])
            count = 0
            average = 0
            i += 1
            continue

    return lst


x_y = ftr(x_y)

"""
a function to remove some of the datapoints
to interpolate them again
 """
def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i,x in enumerate(input_list) if not i in to_delete]

x_y = delete_random_elems(x_y, 81)



new_xy = []
new_z = []

"""
a function to take the coordinates into separte 
list from the values list
"""


def adjust(l):
    for i in range(0, len(x_y) - 1):
        new_xy.append(x_y[i][0])
        new_z.append(x_y[i][1])
        continue

    return new_xy


adjust(x_y)


# function to put each value into list
def extractDigits(lst):
    return [[el] for el in lst]


new_z = extractDigits(new_z)

new_xy = ot.Sample(new_xy)
new_z = ot.Sample(new_z)

# Extract coordinates.
x = np.array(new_xy[:, 0])
y = np.array(new_xy[:, 1])

# Plot the data with a scatter plot and a color map.
fig = plt.figure()
plt.scatter(x, y, c=new_z, cmap='viridis')
plt.colorbar()
plt.show()


"""
a function to write interpolated  points 
into csv file in the form: 
access point, signal, x coordiante, y, floor no.
"""
def write_to_csv(vertices, predictions):
    header = ['ap', 'signal', 'x', 'y', 'z']
    data = []
    for i in range(0, len(predictions) - 1):
        data.append(['A', round(predictions[i][0], 2), round(vertices[i][0], 2), round(vertices[i][1], 2), 1.0])
        i += 1
        continue
    # print(data)
    with open('A1.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


lower = 0.0
upper = 50.0


def fitKriging(x_y, z, isotropic, basis):
    '''
    Fit the parameters of a Kriging metamodel.
    '''
    # Define the Kriging algorithm.
    algo = ot.KrigingAlgorithm(
        x_y, z, isotropic, basis)

    # Set the optimization bounds for the scale parameter to sensible values
    # given the data set.
    scale_dimension = isotropic.getScale().getDimension()
    algo.setOptimizationBounds(ot.Interval([lower] * scale_dimension,
                                           [upper] * scale_dimension))

    # Run the Kriging algorithm and extract the fitted surrogate model.
    algo.run()
    krigingResult = algo.getResult()
    krigingMetamodel = krigingResult.getMetaModel()
    return krigingResult, krigingMetamodel


def plotKrigingPredictions(krigingMetamodel):
    '''
    Plot the predictions of a Kriging metamodel.
    '''
    # Create the mesh of the box [0., 1000.] * [0., 700.]
    myInterval = ot.Interval([0., 0.], [35.0, 45.0])

    # Define the number of intervals in each direction of the box
    nx = 9
    ny = 9
    myIndices = [nx - 1, ny - 1]
    myMesher = ot.IntervalMesher(myIndices)
    myMeshBox = myMesher.build(myInterval)

    # Predict
    vertices = myMeshBox.getVertices()
    predictions = krigingMetamodel(vertices)

    # write to csv

    write_to_csv(vertices, predictions)

    # Format for plot
    X = np.array(vertices[:, 0]).reshape((ny, nx))
    Y = np.array(vertices[:, 1]).reshape((ny, nx))
    predictions_array = np.array(predictions).reshape((ny, nx))

    # Plot
    plt.figure()
    plt.pcolormesh(X, Y, predictions_array, shading='auto')
    plt.title('AP "A" radio map')
    plt.colorbar()
    plt.show()
    return


inputDimension = 2
basis = ot.ConstantBasisFactory(inputDimension).build()
isotropic = ot.IsotropicCovarianceModel(
    ot.SquaredExponential(), inputDimension)
krigingResult, krigingMetamodel = fitKriging(
    new_xy, new_z, isotropic, basis)

plotKrigingPredictions(krigingMetamodel)

"""
inputDimension = 2
basis = ot.ConstantBasisFactory(inputDimension).build()
covarianceModel = ot.SquaredExponential([1.] * inputDimension, [1.0])
algo = ot.KrigingAlgorithm(new_xy, new_z, covarianceModel, basis)
algo.run()
result = algo.getResult()
krigingMetamodel = result.getMetaModel()
print(krigingMetamodel)
plotKrigingPredictions(krigingMetamodel)

test = [25.0, 10.0]
res = krigingMetamodel(test)
print(res)
"""
