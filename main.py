import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# create ridge, lasso and random forest and compare them
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


def categoricalAndPriceRelations(df: pd.DataFrame, column):
    sb.boxplot(x=column, y='selling_price', data=df)
    plt.title(f"Odnos {column} i cene")
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
 
    columns = ["fuel", "seller_type", "transmission", "owner"]
    for column in columns:
        categoricalAndPriceRelations(df, column)


def deleteOrInsertMissedValues(df: pd.DataFrame) -> pd.DataFrame:
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return df


def normalizationOfNumericalArguments(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    numericalColumns = ["year", "km_driven"]

    df[numericalColumns] = scaler.fit_transform(df[numericalColumns])

    return df


def oneHotEncoding(df: pd.DataFrame) -> tuple[str, int]:
    df.drop('name', axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    x = df.drop(columns='selling_price', axis=1)
    y = df['selling_price']

    return (x, y)


def dataPreprocessing(df: pd.DataFrame):
    df = deleteOrInsertMissedValues(df)
    df = normalizationOfNumericalArguments(df)
    x, y = oneHotEncoding(df)

    return (df, x, y)


def convenienceCheck(df: pd.DataFrame):
    numericalColumns = df.select_dtypes(include=['int64']).columns.drop('selling_price')

    for column in numericalColumns:
        plt.scatter(df[column], df['selling_price'])
        plt.title(f"Povera linearnosti: {column} - selling_price")
        plt.xlabel(column)
        plt.ylabel("selling_price")
        plt.grid(True)
        plt.show()


def distributionOfResiduals(x, y):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=0.8, random_state=42)

    model = LinearRegression()
    model.fit(xTrain, yTrain)

    yPred = model.predict(xTest)

    residuals = yTest - yPred

    sb.histplot(residuals, kde=True)
    plt.title("Distribucija reziduala")
    plt.grid(True)
    plt.show()


def evaluateModel(model: LinearRegression, x, y, name):
    yPred = model.predict(x)

    rSquared = r2_score(y, yPred)
    meanSquaredError = mean_squared_error(y, yPred)
    rmse = np.sqrt(meanSquaredError)

    print(f"{name} -> R²: {rSquared:.2f}, MSE: {meanSquaredError:.2f}, RMSE: {rmse:.2f}")


def multipleLinearRegression(df: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame):
    xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, train_size=0.6, random_state=42)
    xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, train_size=0.5, random_state=42)

    model = LinearRegression()
    model.fit(xTrain, yTrain)

    dataSets = [(xTrain, yTrain, "Training"), (xVal, yVal, "Validation"), (xTest, yTest, "Test")]
    for i in range(len(dataSets)):
        dataSet = dataSets[i]
        evaluateModel(model, dataSet[0], dataSet[1], dataSet[2])

    coefficients = pd.DataFrame({
        "Feature": x.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)
    print(coefficients)


def loadCarDetails(fileName) -> pd.DataFrame:
    df = pd.read_csv(fileName)
    return df


if __name__ == "__main__":
    df = loadCarDetails("resources/CAR DETAILS FROM CAR DEKHO.csv")

    exploratoryDataAnalysis(df)
    convenienceCheck(df)

    df, x, y = dataPreprocessing(df)
    distributionOfResiduals(x, y)

    multipleLinearRegression(df, x, y)