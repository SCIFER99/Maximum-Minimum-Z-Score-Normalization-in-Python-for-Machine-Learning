# By: Tim Tarver also known as CryptoKeyPlayer

# This script was created to demonstrate how to
# create more aggregations specified to Maximum,
# Minimum and Z-Score Visual Normalization Data.

# Best seen in Jupyter Notebooks; one function per cell.

# Import the packages we need

import pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# First, we begin with opening the file of data we want to create statistical
# measures with.

def open_csv_file():

    vehicles = pandas.read_csv("vehicles.csv")
    vehicles.head()
    return vehicles

print(open_csv_file())

# Second, we move on to only highlighting descriptions of a specific column (co2emissions)
# and plotting the count, mean, standard deviation, minimum value, maximum value
# and quartiles via histogram.

def statistical_description_of_co2emissions():

    vehicles = pandas.read_csv("vehicles.csv")
    description_table = vehicles[['co2emissions']].describe()
    description_graph = vehicles[['co2emissions']].plot(kind = 'hist',
                                                        bins = 20,
                                                        figsize = (10, 6))
    return vehicles, description_table, description_graph

print(statistical_description_of_co2emissions())

# Now, we want to perform Minimum and Maximum Normalization by scaling our data
# for max and min purposes only and yield new maximum, minimum, count, standard deviation
# mean and quartile values.

def min_max_normalization():

    vehicles = pandas.read_csv("vehicles.csv")
    min_max_co2emissions = MinMaxScaler().fit_transform(vehicles[['co2emissions']])
    min2_max2_co2emissions = pandas.DataFrame(min_max_co2emissions, columns = ['co2emissions'])
    new_description_table = min2_max2_co2emissions.describe()
    new_description_graph = min2_max2_co2emissions.plot(kind = 'hist',
                                                        bins = 20,
                                                        figsize = (10, 6))
    return vehicles, new_description_table, new_description_graph

print(min_max_normalization())

# Finally, we end with the Z-Score Normalization to analyze the data
# in scientific notation.

def z_score_normalization():

    vehicles = pandas.read_csv("vehicles.csv")
    z_score_scaler = StandardScaler().fit_transform(vehicles[['co2emissions']])
    z_score_scaler2 = pandas.DataFrame(z_score_scaler, columns = ['co2emissions'])
    z_score_scaler3 = z_score_scaler2.describe()
    z_score_scaler4 = z_score_scaler3.plot(kind = 'hist',
                         bins = 20,
                         figsize = (10, 6))
    return vehicles, z_score_scaler3, z_score_scaler4

print(z_score_normalization())


    





