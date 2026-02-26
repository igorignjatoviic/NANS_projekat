import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def detectionOfExtremeValues(df: pd.DataFrame, columns):
    for column in columns:
        plt.boxplot(df[column])
        plt.title(f"Detection of extreme values - {column}")
        plt.grid(True)
        plt.show()    


def correlationAnalysis(df: pd.DataFrame):
    corrMatrix = df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sb.heatmap(corrMatrix, annot=True, cmap="coolwarm")
    plt.title("Correlations matrix")
    plt.show()


def categoricalAndPriceRelations(df: pd.DataFrame, column):
    plt.figure(figsize=(8, 6))
    sb.boxplot(x=column, y='selling_price', data=df)
    plt.title(f"{column} - selling_price")
    plt.grid(True)
    plt.show()


def distributionOfNumericalArguments(df: pd.DataFrame):
    numericalColumns = df.select_dtypes(include=['int64']).columns

    for column in numericalColumns:
        plt.hist(df[column], bins=30)
        plt.title(f"Distribution - {column}")
        plt.grid(True)
        plt.show()

    detectionOfExtremeValues(df, numericalColumns)


def exploratoryDataAnalysis(df: pd.DataFrame):
    print(f"Dimensions of dataset: {df.shape}\n")
    print(f"Data types:\n{df.dtypes}\n")
    print(f"Missed values:\n{df.isnull().sum()}\n")

    distributionOfNumericalArguments(df)
    correlationAnalysis(df)
 
    columns = ["fuel", "seller_type", "transmission", "owner"]
    for column in columns:
        categoricalAndPriceRelations(df, column)


def handleMissedValues(df: pd.DataFrame) -> pd.DataFrame:
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return df


def oneHotEncoding(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df.drop('name', axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    x = df.drop(columns='selling_price', axis=1)
    y = df['selling_price']

    return (x, y)


def dataPreprocessing(df: pd.DataFrame):
    df = handleMissedValues(df)
    x, y = oneHotEncoding(df)

    return (df, x, y)


def convenienceCheck(df: pd.DataFrame):
    numericalColumns = df.select_dtypes(include=['int64']).columns.drop('selling_price')

    for column in numericalColumns:
        plt.scatter(df[column], df['selling_price'])
        plt.title(f"Check linearity: {column} - selling_price")
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
    plt.title("Distribution of residuals")
    plt.grid(True)
    plt.show()


def evaluateModel(model: LinearRegression, x, y, name):
    yPred = model.predict(x)

    rSquared, meanSquaredError, rmse = calculateMetrics(y, yPred)

    print(f"{name} -> R²: {rSquared:.2f}, MSE: {meanSquaredError:.2f}, RMSE: {rmse:.2f}")


def multipleLinearRegression(df: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame):
    xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, train_size=0.6, random_state=42)
    xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, train_size=0.5, random_state=42)

    scaler = StandardScaler()
    numericalColumns = ["year", "km_driven"]

    scaler.fit(xTrain[numericalColumns])

    xTrain[numericalColumns] = scaler.transform(xTrain[numericalColumns])
    xVal[numericalColumns] = scaler.transform(xVal[numericalColumns])
    xTest[numericalColumns] = scaler.transform(xTest[numericalColumns])

    model = LinearRegression()
    model.fit(xTrain, yTrain)

    dataSets = [(xTrain, yTrain, "Training"), (xVal, yVal, "Validation"), (xTest, yTest, "Test")]
    for i in range(len(dataSets)):
        dataSet = dataSets[i]
        evaluateModel(model, dataSet[0], dataSet[1], dataSet[2])

    coefficients = calculateCoefficients(xTrain, model)
    print(f"\nCoefficients - Multiple Linear Regression\n", coefficients)


def compareRegressionModels(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, train_size=0.6, random_state=42)
    _, xTest, _, yTest = train_test_split(xTemp, yTemp, train_size=0.5, random_state=42)

    scaler = StandardScaler()
    numericalColumns = ["year", "km_driven"]

    scaler.fit(xTrain[numericalColumns])

    xTrain[numericalColumns] = scaler.transform(xTrain[numericalColumns])
    xTest[numericalColumns] = scaler.transform(xTest[numericalColumns])

    models = generateModels()

    results = []

    for name, model in models.items():
        model.fit(xTrain, yTrain)
        yPred = model.predict(xTest)

        rSqaured, meanSquaredError, rmse = calculateMetrics(yTest, yPred)

        results.append((name, rSqaured, meanSquaredError, rmse))

        if name != "Random Forest":
            coefficients = calculateCoefficients(xTrain, model)
            print(f"\nCoefficients - {name}\n", coefficients)
            plotCoefficients(coefficients, name)
        else:
            importances = calculateImportances(xTrain, model)
            print(f"\nImportances - Random Forest\n", importances)
    
    resultsDf = pd.DataFrame(results, columns=['Model', 'R²', 'MSE', 'RMSE'])

    return resultsDf


def plotModelResults(df: pd.DataFrame):
    plt.figure()
    plt.bar(df['Model'], df['R²'])
    plt.title("R² Comparison")
    plt.show()

    plt.figure()
    plt.bar(df['Model'], df['RMSE'])
    plt.title("RMSE Comparison")
    plt.show()


def plotCoefficients(coefficients, name):
    plt.figure(figsize=(8, 7))
    plt.bar(coefficients['Feature'], coefficients['Coefficient'])
    plt.xticks(rotation=60, fontsize=5)
    plt.title(f"Coefficients - {name}")
    plt.ylabel("Value")
    plt.show()


def generateModels() -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    return models


def calculateCoefficients(x: pd.DataFrame, model) -> pd.DataFrame:
    coefficients = pd.DataFrame({
        "Feature": x.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    return coefficients


def calculateImportances(x: pd.DataFrame, model) -> pd.DataFrame:
    importances = pd.DataFrame({
        "Feature": x.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return importances


def calculateMetrics(stY: pd.DataFrame, ndY: pd.DataFrame) -> tuple[float, float, float]:
    rSqaured = r2_score(stY, ndY)
    meanSqauredError = mean_squared_error(stY, ndY)
    rmse = np.sqrt(meanSqauredError)

    return (rSqaured, meanSqauredError, rmse)


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

    resultsDf = compareRegressionModels(x, y)
    print(f"\nModel comparisson:\n{resultsDf}")
    plotModelResults(resultsDf)