from typing import List, Dict
import numpy as np

class LargestDistanceActiveUserSelector:
    """According to the distance between global module and local module to select client"""

    def get_user_indices(self, kwargs: Dict) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round", "select_percentage", "client_local_model", "global_model", "epoch_num"]
        num_total_users, users_per_round, select_percentage, client_local_model, global_model, round_num = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        total_elements = int(select_percentage * num_total_users)
        user_indices = []
        self._user_indices_overselected = []
        global_model_array = np.frombuffer(global_model.tensors[0], dtype=np.float32)

        if len(client_local_model) == 0:
            for i in range(num_total_users):
                user_indices.append(i)
                self._user_indices_overselected = user_indices
        else:
            clients_distance = []
            for client_model in client_local_model:
                for client_id, parameters in client_model.items():
                    client_model_array = np.frombuffer(parameters.tensors[0], dtype=np.float32)
                    # print(f"Client ID: {client_id}")
                    # print(f"Client Model Array: {client_model_array}")
                
                    global_model_array = global_model_array.reshape(-1)
                    client_model_array = client_model_array.reshape(-1)

                    # Calculate the distance
                    distance = global_model_array - client_model_array

                    # Calculate the Frobenius norm
                    frobenius_norm = np.linalg.norm(distance)
                    clients_distance.append((client_id, frobenius_norm))

            sorted_clients_distance = sorted(clients_distance, key=lambda x: x[1], reverse=True)
            with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist_with_loss_rate_selection/plot/total_sorted_largest_client_distance.txt", 'a') as file:
                for client, distance in sorted_clients_distance:
                    client_norm_info = "Global_round: {}, Client: {}, distance: {}\n".format(round_num, client, distance)
                    file.write(client_norm_info)

            with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist_with_loss_rate_selection/plot/selected_sorted_largest_client_distance.txt", 'a') as file:
                for client, distance in sorted_clients_distance[0:total_elements]:
                    self._user_indices_overselected.append(client)
               
                    selected_client_info = "Global_round: {}, Client: {}, distance: {}\n".format(round_num, client, distance)
                    file.write(selected_client_info)

        print(f"client index: {self._user_indices_overselected}")
        with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist_with_loss_rate_selection/plot/selected_clients.txt", "a") as file:
                file.write(f"{self._user_indices_overselected}\n")

        return self._user_indices_overselected

    def unpack_required_inputs(self, required_inputs, kwargs):
        return [kwargs[input_name] for input_name in required_inputs]