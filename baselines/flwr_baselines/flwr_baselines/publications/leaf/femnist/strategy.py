"""FedAvg with the same clients used for both training and evaluation."""

from typing import Callable, Dict, List, Optional, Tuple,Union

from flwr.common import (
    EvaluateIns,
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
     ndarrays_to_parameters,
    FitRes,
    EvaluateRes,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.logger import log
import random
import logging

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

WARNING = logging.WARNING

class FedAvgSameClients(FedAvg):
    """FedAvg that samples clients for each round only once (the same clients
    are used for training and testing round n)

    It does not mean that the same client are used in each round. It used just the same clients
    (with different parts of their data) in round i.

    It assumes that there is no different function for evaluation - on_evaluate_config_fn
    (it's ignored).
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self._current_round_fit_clients_fits_list: List[Tuple[ClientProxy, FitIns]] = []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self._current_round_fit_clients_fits_list = super().configure_fit(
            server_round, parameters, client_manager
        )

    # Extract the selected client IDs
        selected_client_ids = [client.cid for client, _ in self._current_round_fit_clients_fits_list]
        
        # Print the round number and selected client IDs in the required format
        print(f"{server_round}: {selected_client_ids}")

        # Format the output
        output_line = f"{server_round}: {selected_client_ids}\n"

        with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/client_selection.txt", "a") as file:  # Open in append mode to keep adding rounds
            file.write(output_line)
    

        # Return client/config pairs
        return self._current_round_fit_clients_fits_list

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Keep the fraction_settings for consistency reasons
        if self.fraction_evaluate == 0.0:
            return []
        evaluate_config = []
        for tuple_client_proxy_fit_ins in self._current_round_fit_clients_fits_list:
            eval_ins = EvaluateIns(
                tuple_client_proxy_fit_ins[1].parameters,
                tuple_client_proxy_fit_ins[1].config,
            )
            evaluate_config.append((tuple_client_proxy_fit_ins[0], eval_ins))
        return evaluate_config
    


class include_or_exclude_clients_random(FedAvg):
    """Configurable FedAvg strategy implementation with random client selection."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        include_clients_0_and_1: bool = True,
        num_inclusive_exclusive_clients: float = 2.0,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
    ) -> None:

        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, "WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW")
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.include_clients_0_and_1 = include_clients_0_and_1
        self.num_inclusive_exclusive_clients = num_inclusive_exclusive_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"RandomSelect(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        # Initialize an empty configuration dictionary
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Get a list of all clients managed by the client_manager
        all_clients = list(client_manager.all().values())
        selected_client_proxies = []
        
        # If the flag is set to include clients "0" and "1" by default, add them to the selected list
        # if self.include_clients_0_and_1:
        #     # Add clients with IDs "0" and "1" to selected list
        #     selected_client_proxies.extend(
        #         [client_manager.clients.get(cid) for cid in ["0", "1"] if cid in client_manager.clients]
        #     )
        #     remaining_clients = [client for client in all_clients if client.cid not in ["0", "1"]]
        # else:
        #     remaining_clients = [client for client in all_clients if client.cid not in ["0", "1"]]

        # num_clients_to_select = int(self.min_fit_clients) - len(selected_client_proxies)
        # selected_client_proxies.extend(random.sample(remaining_clients, num_clients_to_select))
        inclusive_client_ids = [str(i) for i in range(self.num_inclusive_exclusive_clients)]
        if self.include_clients_0_and_1:
            # Generate the list of client IDs to include based on the parameter
            # Add clients with the generated IDs to the selected list
            selected_client_proxies.extend(
                [client_manager.clients.get(cid) for cid in inclusive_client_ids if cid in client_manager.clients]
            )
            remaining_clients = [client for client in all_clients if client.cid not in inclusive_client_ids]
        else:
            remaining_clients = [client for client in all_clients if client.cid not in inclusive_client_ids]
        
        num_clients_to_select = int(self.min_fit_clients) - len(selected_client_proxies)
        selected_client_proxies.extend(random.sample(remaining_clients, num_clients_to_select))

        # Ensure enough clients are available
        if len(selected_client_proxies) < self.min_fit_clients:
            print(f"Not enough clients for training. Required: {self.min_fit_clients}, Available: {len(selected_client_proxies)}")
            return []

        # Print the selected clients for the next round
        selected_client_ids = [client.cid for client in selected_client_proxies if client is not None]
        print(f"Selected clients for round {server_round}: {selected_client_ids}")

        # Save the selected clients for the next round to the specified file
        with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/client_selection.txt", "a") as file:
            file.write(f"{server_round}: {selected_client_ids}\n")

        return [(client, fit_ins) for client in selected_client_proxies if client is not None]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated