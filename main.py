import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# autocorrelations analysis
def detectionOfExtremeValues(df: pd.DataFrame, columns):
    for column in columns:
        plt.boxplot(df[column])
        plt.title(f"Detekcija ekstremnih vrednosti - {column}")
        plt.grid(True)
        plt.show()    


def distributionOfNumericalArguments(df: pd.DataFrame):
    numericalColumns = df.select_dtypes(include=['int64']).columns

    for column in numericalColumns:
        plt.hist(df[column], bins=30)
        plt.title(f"Distribucija - {column}")
        plt.grid(True)
        plt.show()

    detectionOfExtremeValues(df, numericalColumns)


def exploratoryDataAnalysis(df: pd.DataFrame):
    print(f"Dimenzije skupa podatka: {df.shape}\n")
    print(f"Tipovi podataka:\n{df.dtypes}\n")
    print(f"Nedostajuce vrednosti:\n{df.isnull().sum()}\n")

    distributionOfNumericalArguments(df)

    
def loadCarDetails(fileName):
    df = pd.read_csv(fileName)
    return df


if __name__ == "__main__":
    df = loadCarDetails("resources/CAR DETAILS FROM CAR DEKHO.csv")
    exploratoryDataAnalysis(df)