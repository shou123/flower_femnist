"""FedAvg with the same clients used for both training and evaluation."""

from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr_baselines.publications.leaf.femnist_with_large_distance_selection.selector import LargestDistanceActiveUserSelector


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
        client_selection_fn: Optional[Callable[[Dict], List[int]]] = None,
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
        self.client_selection_fn = client_selection_fn
        self.client_local_model: List[Dict[int, NDArrays]] = []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # self._current_round_fit_clients_fits_list = super().configure_fit(
        #     server_round, parameters, client_manager
        # )

        if self.client_selection_fn:
            selected_client_ids = self.client_selection_fn(
                {
                    "num_total_users": len(client_manager.all()),
                    "users_per_round": self.min_fit_clients,
                    "select_percentage": 0.8,
                    "client_local_model": self.client_local_model,
                    "global_model": parameters,
                    "epoch_num": server_round,
                }
            )
            print(f"Selected client IDs: {selected_client_ids}")

            # Ensure selected_client_ids is defined
            if selected_client_ids is None:
                raise ValueError("client_selection_fn did not return any client IDs")

            # Get all clients
            all_clients = client_manager.all()
            print(f"All clients: {all_clients}")

            # Filter selected clients
            selected_clients = []
            for cid, client in all_clients.items():
                if int(cid) in selected_client_ids:
                    selected_clients.append(client)
            print(f"Selected clients: {selected_clients}")

            # Create the list of client/config pairs
            self._current_round_fit_clients_fits_list = []
            for client in selected_clients:
                fit_ins = FitIns(parameters, {})
                self._current_round_fit_clients_fits_list.append((client, fit_ins))
            # print(f"Current round fit clients: {self._current_round_fit_clients_fits_list}")
        else:
            self._current_round_fit_clients_fits_list = super().configure_fit(
                server_round, parameters, client_manager
            )
        # Return client/config pairs
        return self._current_round_fit_clients_fits_list
    

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        # Update client_local_model with the latest parameters
        for client_proxy, fit_res in results:
            client_id = int(client_proxy.cid)
            self.client_local_model.append({client_id: fit_res.parameters})

        return aggregated_parameters, aggregated_metrics

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
