import argparse
import logging
import flwr as fl
import numpy as np
import json
import os
from joblib import dump, load

# Custom Strategy for Logistic Regression
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds=20, **kwargs):
        super().__init__(**kwargs)
        self.metrics_file = "federated_learning/server/metrics/metrics.json"
        self.num_rounds = num_rounds
        self._clear_previous_metrics()

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results from clients and log global parameters."""
        logging.info(f"Aggregating fit results for round {server_round}.")
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters:
            # Log the aggregated global parameters
            logging.debug(f"Aggregated parameters after round {server_round}: {aggregated_parameters}")
            if server_round == self.num_rounds:
                self._save_global_model(aggregated_parameters)
        else:
            logging.warning(f"No aggregated parameters for round {server_round}.")
        
        return aggregated_parameters

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics from clients."""
        valid_results = [result.metrics for _, result in results if result.metrics]
        if not valid_results:
            logging.error(f"No valid metrics received in round {server_round}.")
            return 0.0, {}

        aggregated_metrics = self._aggregate_metrics(valid_results)
        losses = [result.loss for _, result in results if result.loss is not None]
        average_loss = np.mean(losses) if losses else 0.0

        self._save_metrics_to_json(server_round, average_loss, aggregated_metrics)
        return average_loss, aggregated_metrics
    
    def _save_global_model(self, global_parameters, output_path="federated_learning/server/models/global_model.pkl"):
        """Save the final global model parameters."""
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save using Joblib
            dump(global_parameters, output_path)
            logging.info(f"Global model saved at {output_path}.")
        except Exception as e:
            logging.error(f"Error saving the global model: {e}")

    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics across all clients."""
        return {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0].keys()}

    def _save_metrics_to_json(self, round_num, loss, metrics):
        """Save metrics to a JSON file."""
        data = {"rounds": []}
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON format in {self.metrics_file}. Overwriting.")

        # Append the new round's data
        data["rounds"].append({"round": round_num, "loss": loss, "metrics": metrics})
        
        # Save the updated data
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(data, f, indent=4)
            logging.info(f"Saved metrics for round {round_num} to {self.metrics_file}.")
        except IOError as e:
            logging.error(f"Error saving metrics to {self.metrics_file}: {e}")

    def _clear_previous_metrics(self):
        """Delete the metrics file at server startup."""
        if os.path.exists(self.metrics_file):
            try:
                os.remove(self.metrics_file)
                logging.info(f"Deleted metrics file: {self.metrics_file}.")
            except Exception as e:
                logging.error(f"Error deleting metrics file: {e}")
        else:
            logging.info(f"No metrics file found at {self.metrics_file}. Starting fresh.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated learning rounds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Use the custom strategy for Logistic Regression
    strategy = CustomFedAvg(num_rounds=args.num_rounds, fraction_fit=1.0, min_available_clients=2)

    fl.server.start_server(strategy=strategy, config=fl.server.ServerConfig(num_rounds=args.num_rounds))