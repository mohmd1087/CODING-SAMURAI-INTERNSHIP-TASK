import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load the dataset
iris = pd.read_csv("Iris.csv")

# Step 2: Data exploration
print(iris.head())
print(iris['Species'].value_counts())

# Step 3: Data preprocessing
# sns.FacetGrid(iris, hue="Species", height=8).map(plt.scatter, "PetalLengthCm", "SepalWidthCm").add_legend()

flower_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris["Species"] = iris["Species"].map(flower_mapping)

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = iris[['Species']].values

# Step 4: Create a pipeline with preprocessing and Logistic Regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('model', LogisticRegression())  # Logistic Regression model
])

# Step 5: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Step 6: Save the pipeline (including model) to a pickle file
with open('iris_pipeline_model.pkl', 'wb') as pipeline_file:
    pickle.dump(pipeline, pipeline_file)

# Save the DataFrame to another pickle file
iris.to_pickle('iris_dataframe.pkl')
