from typing import List, Dict

class LossRateActiveUserSelector:
    """Select clients based on their loss values."""

    def get_user_indices(self, kwargs: Dict) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round", "select_percentage", "client_losses", "epoch_num"]
        num_total_users, users_per_round, select_percentage, client_losses, round_num = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        total_elements = int(select_percentage * num_total_users)
        user_indices = []
        self._user_indices_overselected = []

        if len(client_losses) == 0:
            for i in range(num_total_users):
                user_indices.append(i)
                self._user_indices_overselected = user_indices
        else:
            clients_loss = [(client_id, loss) for client_id, loss in client_losses.items()]

            sorted_clients_loss = sorted(clients_loss, key=lambda x: x[1],reverse=True)
            with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist_with_loss_rate_selection/plot/total_sorted_client_loss.txt", 'a') as file:
                for client, loss in sorted_clients_loss:
                    client_loss_info = "Global_round: {}, Client: {}, loss: {}\n".format(round_num, client, loss)
                    file.write(client_loss_info)

            with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist_with_loss_rate_selection/plot/selected_sorted_client_loss.txt", 'a') as file:
                for client, loss in sorted_clients_loss[0:total_elements]:
                    self._user_indices_overselected.append(client)
               
                    selected_client_info = "Global_round: {}, Client: {}, loss: {}\n".format(round_num, client, loss)
                    file.write(selected_client_info)

        print(f"client index: {self._user_indices_overselected}")
        with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist_with_loss_rate_selection/plot/selected_clients.txt", "a") as file:
                file.write(f"{self._user_indices_overselected}\n")

        return self._user_indices_overselected

    def unpack_required_inputs(self, required_inputs, kwargs):
        return [kwargs[input_name] for input_name in required_inputs]