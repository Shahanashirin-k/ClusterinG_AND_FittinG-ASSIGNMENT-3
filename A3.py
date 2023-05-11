# -*- coding: utf-  -*-
"""
Created on Tue May  2 19:17:18 2023

@author: SHAHANA SHIRIN
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import errors as err
import cluster_tools as ct


def read_clean(fn):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    fn (str): The filename of the CSV file to be read.

    Returns:
    df (pandas.DatFrame): The DataFrame containing the data read from the CSV file.
    """

    # Read the data file frome the adress in to a dataframe
    address = "C:/Users/SHAHANA SHIRIN/Desktop/ADS/assignment3/" + fn
    df = pd.read_csv(address)

    #Cleaning the dataframe for getting the specific country and its data
    df = df.drop(df.columns[:2], axis=1)
    df = df.drop(columns=['Country Code'])

    # Remove the string of year from column names,Ecxtracting the data transposing it and renaming it.
    df.columns = df.columns.str.replace(' \[YR\d{4}\]', '', regex=True)
    countries = ['China', 'India']
    country_code = ['CHN', 'USA']
    df = df[df['Country Name'].isin(countries)].transpose()
    df = df.rename({'Country Name': 'year'})
    df = df.reset_index().rename(columns={'index': 'year'})
    df.columns = df.iloc[0]
    df = df.iloc[1:]

    # Replace missing values with 0
    df = df.replace(np.nan, 0)

    # Convert data types to the correct format
    df["year"] = df["year"].astype(int)
    df['India'] = df['India'].astype(float)
    df['China'] = df['China'].astype(float)
    return df


def curve_fun(t, scale, growth):
    """
    Calculates the value of a function with exponential growth.

    Parameters:
    t (int): The time variable for which to calculate the function value.
    scale (float): The starting value of the function.
    growth (float): The rate of exponential growth.

    Returns:
    f (float): The value of the function at the given time.
    """
    f = scale * np.exp(growth * (t-1990))
    return f


def plot_gdp(df_gdp):
    """
    Plots the GDP per capita for China and India over the years.

    Parameters:
    df_gdp (pandas.DataFrame): A DataFrame containing the GDP per capita data.

    Returns:
    None.
    """
    plt.plot(df_gdp["year"], df_gdp["China"], label='China')
    plt.plot(df_gdp["year"], df_gdp['India'], label='India')
    plt.xlim(1990, 2019)
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("GDP Per Capita", fontsize=15)
    plt.legend(fontsize=15)
    plt.title("GDP per capita", fontsize=15)
    plt.savefig("GDP.png", dpi=300, bbox_inches='tight')
    plt.show()


def curve_fit_and_plot(df, country):
    """
    Fits a curve to the CO2 emissions data for the specified country and plots
    both the data and the fitted curve.

    Returns:
        tuple: A tuple containing the parameters and covariance matrix of the fitted curve.
    """
    x = df['year']
    y = df[country]
    param, cov = opt.curve_fit(curve_fun, x, y, p0=[4e8, 0.1])
    sigma = np.sqrt(np.diag(cov))
    low, up = err.err_ranges(x, curve_fun, param, sigma)
    df["fit_value"] = curve_fun(x, * param)
    # Plotting the figure
    plt.figure()
    plt.title(f"{country} CO2 emissions (metric tons per capita)", fontsize=15)
    plt.plot(x, y, label="data", c="green")
    plt.plot(x, df["fit_value"], c="deeppink", label="fit")
    plt.fill_between(x, low, up, alpha=0.2)
    plt.legend(fontsize=15)
    plt.xlim(1990, 2019)
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("CO2 Emission(mt/c)", fontsize=15)
    plt.savefig(f"{country}_curve_fit.png", dpi=300, bbox_inches='tight')
    plt.show()
    # Return parameters and covariance matrix
    return param, cov


def predict_co2_emission(df_co2, country):
    """
    Generates a plot of predicted CO2 emissions for a given country.
    """
    # Fit the curve to the data and generate predicted values
    param, cov = opt.curve_fit(
        curve_fun, df_co2["year"], df_co2[country], p0=[4e8, 0.1])
    sigma = np.sqrt(np.diag(cov))
    low, up = err.err_ranges(df_co2["year"], curve_fun, param, sigma)
    df_co2["fit_value"] = curve_fun(df_co2["year"], *param)

    # Plotting the predicted values
    plt.figure()
    plt.title(f"{country} CO2 emission prediction", fontsize=15)
    pred_year = np.arange(1990, 2030)
    pred_ind = curve_fun(pred_year, *param)
    plt.plot(df_co2["year"], df_co2[country], label="data", c="green")
    plt.plot(pred_year, pred_ind, label="predicted values",
             c="red", linestyle="--")
    plt.legend(fontsize=15)
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("CO2", fontsize=15)
    plt.savefig(f"{country}_prediction.png", dpi=300, bbox_inches='tight')
    plt.show()


def predict_renewable_energy(df_renew, country):
    """
    Generates a plot of predicted CO2 emissions for a given country.
    """
    # Fit the curve to the data and generate predicted values
    param, cov = opt.curve_fit(
        curve_fun, df_renew["year"], df_renew[country], p0=[4e8, 0.1])
    sigma = np.sqrt(np.diag(cov))
    low, up = err.err_ranges(df_renew["year"], curve_fun, param, sigma)
    df_renew["fit_value"] = curve_fun(df_renew["year"], *param)

    # Plotting the predicted values
    plt.figure()
    plt.title(f"{country} Renewable Energy prediction", fontsize=15)
    pred_year = np.arange(1990, 2030)
    pred_ind = curve_fun(pred_year, *param)
    plt.plot(df_renew["year"], df_renew[country], label="data", c="green")
    plt.plot(pred_year, pred_ind, label="predicted values",
             c="red", linestyle="--")
    plt.legend(fontsize=15)
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Renewable Energy(%)", fontsize=15)
    plt.savefig(f"{country}_renewable_energy.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def clustering(df_co2, df_renew):
    """
    Generates a scatter plot and saves it as a PNG file.
    """
    # Scale the data using the scaler function from ct
    df_norm, df_min, df_max = ct.scaler(df_co2)

    # Create cluster and visualize CO2 emissions of given countries
    kmean = cluster.KMeans(n_clusters=4).fit(df_norm)
    label = kmean.labels_
    plt.scatter(df_norm["China"], df_norm["India"], c=label, cmap="tab10")
    plt.xlabel("India CO2 Emission", fontsize=15)
    plt.ylabel("China CO2 Emission", fontsize=15)
    plt.title("UK and India - CO2 Emission", fontsize=15)
    c = kmean.cluster_centers_

    # Back scaling
    c = ct.backscale(c, df_min, df_max)
    x = c[:, 0]
    y = c[:, 1]
    plt.savefig("norm.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create new DataFrame for the given country and cluster and visualize CO2 emissions vs renewable energy usage
    df_India = pd.DataFrame()
    df_India["co2_emission"] = df_co2['India']
    df_India["renew_energy"] = df_renew['India']
    kmean = cluster.KMeans(n_clusters=4).fit(df_India)
    label = kmean.labels_
    plt.scatter(df_India["co2_emission"],
                df_India["renew_energy"], c=label, cmap="tab10")
    plt.title(f"CO2 Emission vs Renewable Energy Usage-India", fontsize=15)
    plt.xlabel("CO2 Emission (mt/capita)", fontsize=15)
    plt.ylabel("Renewable Energy(%)", fontsize=15)
    c = kmean.cluster_centers_
    for t in range(4):
        xc, yc = c[t, :]
        plt.plot(xc, yc, "xk", markersize=8)
    plt.savefig("co2_ener.png", dpi=300, bbox_inches='tight')
    plt.show()


# main function for visualization:
if __name__ == '__main__':
    """
    Main function that calls the other functions to create the plots.
    """
    # calling the read functions in to a new dataframe
    df_co2 = read_clean('co2pc.csv')

    df_gdp = read_clean('gdpannum.csv')
    df_renew = read_clean('ren_energy.csv')

    # calling gdp plot for visualization
    plot_gdp(df_gdp)

    # Curve fit and plot for China and India
    param_china, cov_china = curve_fit_and_plot(df_co2, "China")
    param_india, cov_india = curve_fit_and_plot(df_co2, "India")

    # Curve fit and plot the predicted CO2 emission and Renewable Energy use for China and India
    predict_co2_emission(df_co2, 'China')
    predict_co2_emission(df_co2, 'India')
    predict_renewable_energy(df_renew, 'India')
    predict_renewable_energy(df_renew, 'China')

    # calling for clustering
    clustering(df_co2, df_renew)
