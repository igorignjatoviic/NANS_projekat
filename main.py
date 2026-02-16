import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# analize the correlations so we could make linear regression
def detectionOfExtremeValues(df: pd.DataFrame, columns):
    for column in columns:
        plt.boxplot(df[column])
        plt.title(f"Detekcija ekstremnih vrednosti - {column}")
        plt.grid(True)
        plt.show()    


def correlationAnalysis(df: pd.DataFrame):
    corrMatrix = df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sb.heatmap(corrMatrix, annot=True, cmap="coolwarm")
    plt.title("Matrica korelacije")
    plt.show()


def categoricalAndPriceRelations(df: pd.DataFrame):
    sb.boxplot(x='fuel', y='selling_price', data=df)
    plt.title("Odnos tipa goriva i cene")
    plt.grid(True)
    plt.show()

    sb.boxplot(x='seller_type', y='selling_price', data=df)
    plt.title("Odnos tipa prodavca i cene")
    plt.grid(True)
    plt.show()

    sb.boxplot(x='transmission', y='selling_price', data=df)
    plt.title("Odnos tipa menjaca i cene")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 7))
    sb.boxplot(x='owner', y='selling_price', data=df)
    plt.title("Odnos vlasnika i cene")
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
    correlationAnalysis(df)
    categoricalAndPriceRelations(df)


def deleteOrInsertMissedValues(df: pd.DataFrame):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return df


def normalizationOfNumericalArguments(df: pd.DataFrame):
    scaler = StandardScaler()
    numericalColumns = ["year", "km_driven"]

    df[numericalColumns] = scaler.fit_transform(df[numericalColumns])

    return df


def oneHotEncoding(df: pd.DataFrame):
    df.drop('name', axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    x = df.drop(columns='selling_price', axis=1)
    y = df['selling_price']

    return (x, y)


def dataPreprocessing(df: pd.DataFrame):
    df = deleteOrInsertMissedValues(df)
    df = normalizationOfNumericalArguments(df)
    oneHotEncoding(df)      # function returns tuple of data

    
def loadCarDetails(fileName):
    df = pd.read_csv(fileName)
    return df


if __name__ == "__main__":
    df = loadCarDetails("resources/CAR DETAILS FROM CAR DEKHO.csv")
    exploratoryDataAnalysis(df)
    dataPreprocessing(df)