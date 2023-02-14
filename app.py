from flask import Flask
app = Flask(__name__)

from explainerdashboard import RegressionExplainer, ExplainerDashboard
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
data = load_diabetes()
X = pd.DataFrame(data.data, columns = data.feature_names)
y=pd.DataFrame(data.target,columns=["target"])
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
model = RandomForestRegressor(n_estimators=50, max_depth=5)
model.fit(X_train, y_train.values.ravel())

explainer = RegressionExplainer(model, X_test, y_test)

db = ExplainerDashboard(explainer, 
                        title="Diabetes Prediction", # defaults to "Model Explainer"
                        whatif=False
                        
                        )


@app.route('/')
def hello_world():
    db.app.index()
