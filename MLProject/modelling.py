import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)
from mlflow.models.signature import infer_signature
import os

# Setup
random_state = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "train_data.csv")
test_path = os.path.join(BASE_DIR, "test_data.csv")
model_artifact_path = "model"

# Load pre-split train and test sets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["Heating_Load", "Cooling_Load"])
y_train = train_df[["Heating_Load", "Cooling_Load"]]
X_test = test_df.drop(columns=["Heating_Load", "Cooling_Load"])
y_test = test_df[["Heating_Load", "Cooling_Load"]]

def train_and_log_model(run_name_suffix, tracking_uri=None):
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    
    with mlflow.start_run(run_name=f"RandomForest_MultiOutput_{run_name_suffix}"):
        # Model
        base_model = RandomForestRegressor(random_state=random_state)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Log parameters
        mlflow.log_param("model_type", "MultiOutputRandomForest")
        mlflow.log_param("input_features", X_train.shape[1])
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Log multiple evaluation metrics per target
        for i, target in enumerate(["Heating_Load", "Cooling_Load"]):
            y_true = y_test.iloc[:, i]
            y_pred_i = y_pred[:, i]

            rmse = mean_squared_error(y_true, y_pred_i, squared=False)
            r2 = r2_score(y_true, y_pred_i)
            mae = mean_absolute_error(y_true, y_pred_i)
            mape = mean_absolute_percentage_error(y_true, y_pred_i)
            evs = explained_variance_score(y_true, y_pred_i)

            mlflow.log_metric(f"rmse_test_{target}", rmse)
            mlflow.log_metric(f"r2_test_{target}", r2)
            mlflow.log_metric(f"mae_test_{target}", mae)
            mlflow.log_metric(f"mape_test_{target}", mape)
            mlflow.log_metric(f"explained_var_test_{target}", evs)
        
        # Log model with input example and signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, y_pred)
        
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_artifact_path,
            input_example=input_example,
            signature=signature
        )

if __name__ == "__main__":
    train_and_log_model(run_name_suffix="modelling")