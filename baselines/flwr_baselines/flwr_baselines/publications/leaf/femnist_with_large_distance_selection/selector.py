from typing import List, Dict

class LargestDistanceActiveUserSelector:
    """According to the distance between global module and local module to select client"""

    # def __init__(self, **kwargs):
    #     self.cur_round_user_index = 0

    # @classmethod
    # def _set_defaults_in_cfg(cls, cfg):
    #     pass

    def get_user_indices(self, kwargs: Dict) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round", "select_percentage", "client_local_model", "global_model", "epoch_num"]
        num_total_users, users_per_round, select_percentage, client_local_model, global_model, epoch_num = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        total_elements = int(select_percentage * num_total_users)
        user_indices = []
        self._user_indices_overselected = []

        if len(client_local_model) == 0:
            for i in range(num_total_users):
                user_indices.append(i)
                self._user_indices_overselected = user_indices
        else:
            clients_distance = []
            for client_model in client_local_model:
                for key, value in client_model.items():
                    distance = global_model - value
                    frobenius_norm = distance.norm('fro')
                    clients_distance.append((key, frobenius_norm.item()))

            sorted_clients_distance = sorted(clients_distance, key=lambda x: x[1], reverse=True)
            with open("results/sorted_largest_client_distance.txt", 'a') as file:
                for client, distance in sorted_clients_distance:
                    client_norm_info = "Global_round: {}, Client: {}, distance: {}\n".format(epoch_num, client, distance)
                    file.write(client_norm_info)

            for key, _ in sorted_clients_distance[0:total_elements]:
                self._user_indices_overselected.append(key)

        print(f"client index: {self._user_indices_overselected}")
        return self._user_indices_overselected

    def unpack_required_inputs(self, required_inputs, kwargs):
        return [kwargs[input_name] for input_name in required_inputs]