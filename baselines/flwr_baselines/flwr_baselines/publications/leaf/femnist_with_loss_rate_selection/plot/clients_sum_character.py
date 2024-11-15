from collections import defaultdict

# Initialize a dictionary to store the sums for each client
client_sample_sums = defaultdict(float)

# Define the path to the proportions.txt file
file_path = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist_with_loss_rate_selection/plot/proportions.txt"

# Read the proportions.txt file
with open(file_path, "r") as file:
    for line in file:
        # Parse the line
        parts = line.strip().split(", ")
        if len(parts) == 2:
            client_id = int(parts[0].split(" ")[1])
            proportion_info = parts[1].split(": ")
            character = proportion_info[0].split(" ")[1]
            proportion = float(proportion_info[1])
            
            # Add the proportion to the corresponding client
            client_sample_sums[client_id] += proportion

# Print the total number of samples for each client
for client_id, total_samples in client_sample_sums.items():
    print(f"Client {client_id}: proportion = {total_samples}")