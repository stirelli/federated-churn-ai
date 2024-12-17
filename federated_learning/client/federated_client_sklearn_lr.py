import argparse
import logging
import flwr as fl
import numpy as np
from utils import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

class LogisticRegressionClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dir):
        self.client_id = client_id
        self.data_path = f"{data_dir}/client_{client_id}_data.csv"
        self.X, self.y = self.load_and_preprocess_data()
        self.model = LogisticRegression(C=5, penalty="l2", max_iter=3, warm_start=True)
        self.is_trained = False

    def load_and_preprocess_data(self):
        """Load and preprocess client-specific data."""
        try:
            encoder_path = "federated_learning/client/metadata/encoder.pkl"
            scaler_path = "federated_learning/client/metadata/scaler.pkl"
            logging.info(f"Loading data for client {self.client_id} from {self.data_path}")
            X, y = preprocess_data(self.data_path, encoder_path, scaler_path)

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Store test set for evaluation
            self.X_test = X_test
            self.y_test = y_test

            return X_train, y_train
        except Exception as e:
            logging.error(f"Error loading data for client {self.client_id}: {e}")
            raise e

    def get_parameters(self, config=None):
        """Retrieve model parameters."""
        if not self.is_trained or not hasattr(self.model, "coef_"):
            logging.warning("Model is not trained yet. Sending initial parameters.")
            # Enviar parámetros iniciales con valores cero
            n_features = self.X.shape[1]  # Número de características
            return [np.zeros((1, n_features)).tolist(), [0.0]]  # Coeficientes e intercepto iniciales

        params = [self.model.coef_.tolist(), self.model.intercept_.tolist()]
        logging.info(f"Sending parameters to server: {params}")
        return params

    def set_parameters(self, parameters):
        if parameters is None or len(parameters) == 0:
            logging.warning("No valid parameters received. Proceeding with random initialization.")
            return
        try:
            # Convertir los coeficientes e intercepto a numpy arrays
            self.model.coef_ = np.array(parameters[0])
            self.model.intercept_ = np.array(parameters[1])
            logging.info(f"Client {self.client_id}: Parameters set successfully.")
        except Exception as e:
            logging.error(f"Error setting parameters: {e}")
            logging.warning("Proceeding with random initialization.")

    def fit(self, parameters, config):
        logging.info(f"Client {self.client_id}: Starting training.")
        
        # Recibir y establecer parámetros iniciales
        if parameters:
            logging.info(f"Client {self.client_id}: Received parameters: {parameters}")
            self.set_parameters(parameters)
        
        # Entrenar modelo
        self.model.fit(self.X, self.y)
        self.is_trained = True
        
        # Actualizar parámetros después del entrenamiento
        updated_params = [np.array(self.model.coef_), np.array(self.model.intercept_)]
        logging.info(f"Client {self.client_id}: Parameters after training: {updated_params}")
        
        # Número de muestras y métricas opcionales (si aplica)
        num_samples = len(self.X)
        metrics = {}  # Aquí puedes añadir métricas opcionales
        
        # Asegurarte de devolver los tipos correctos
        return updated_params, num_samples, metrics

    def evaluate(self, parameters, config):
        """Evaluate the model and return metrics."""
        if parameters:
            logging.info(f"Client {self.client_id}: Received parameters for evaluation: {parameters}")
            self.set_parameters(parameters)

        if not self.is_trained or not hasattr(self.model, "coef_"):
            logging.error("Model not trained. Skipping evaluation.")
            return 0.0, 0, {}

        predictions = self.model.predict(self.X_test)
        probabilities = self.model.predict_proba(self.X_test)[:, 1]

        # Log predictions and probabilities for debugging
        logging.debug(f"Client {self.client_id}: Predictions: {predictions}")
        logging.debug(f"Client {self.client_id}: Probabilities: {probabilities}")

        # Metrics
        loss = log_loss(self.y_test, probabilities)
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, zero_division=0)
        recall = recall_score(self.y_test, predictions, zero_division=0)
        f1 = f1_score(self.y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, probabilities)

        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc}
        logging.info(f"Evaluation metrics: {metrics}")
        return loss, len(self.X_test), metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--data_dir", type=str, default="federated_learning/client/data", help="Data directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    client = LogisticRegressionClient(client_id=args.client_id, data_dir=args.data_dir)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)