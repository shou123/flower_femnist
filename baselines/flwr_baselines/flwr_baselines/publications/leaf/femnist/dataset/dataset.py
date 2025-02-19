"""FEMNIST dataset creation module."""

import pathlib
from logging import INFO
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from flwr_baselines.publications.leaf.femnist.dataset.nist_preprocessor import (
    NISTPreprocessor,
)
from flwr_baselines.publications.leaf.femnist.dataset.nist_sampler import NistSampler
from flwr_baselines.publications.leaf.femnist.dataset.zip_downloader import (
    ZipDownloader,
)

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from collections import Counter
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

class NISTLikeDataset(Dataset):
    """Dataset representing NIST or preprocessed variant of it."""

    def __init__(
        self,
        image_paths: List[pathlib.Path],
        labels: np.ndarray,
        transform: transforms = transforms.ToTensor(),
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.ToTensor() if transform is None else transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label


def create_dataset(df_info: pd.DataFrame, labels: np.ndarray) -> NISTLikeDataset:
    """Instantiate NISTLikeDataset.

    Parameters
    ----------
    df_info: pd.DataFrame
        contains paths to images
    labels: np.ndarray
        0 till N-1 classes labels in the same order as in df_info

    Returns
    -------
    nist_like_dataset: NISTLikeDataset
        created dataset
    """
    nist_like_dataset = NISTLikeDataset(df_info["path"].values, labels)
    return nist_like_dataset


def create_partition_list(df_info: pd.DataFrame) -> List[List[int]]:
    """Create list of list with int masks identifying writers.

    Parameters
    ----------
    df_info: pd.DataFrame
        contains writer_id information

    Returns
    -------
    division_list: List[List[int]]
        List of lists of indices to identify unique writers
    """
    writers_ids = df_info["writer_id"].values
    unique_writers = np.unique(writers_ids)
    indices = {
        writer_id: np.where(writers_ids == writer_id)[0].tolist()
        for writer_id in unique_writers
    }
    return list(indices.values())

def partition_dataset(
    dataset: Dataset, division_list: List[List[int]]
) -> List[Dataset]:
    """
    Partition dataset for niid settings - by writer id (each partition has only single writer data).
    Parameters
    ----------
    dataset: Dataset
        dataset of all images
    division_list: List[List[int]]
        list of lists of indices to identify unique writers

    Returns
    -------
    subsets: List[Dataset]
        subsets of datasets divided by writer id
    """
    subsets = []
    for sequence in division_list:
        subsets.append(Subset(dataset, sequence))
    return subsets


# def partition_dataset(
#     dataset: Dataset, partition_indices: List[List[int]]
# ) -> List[Dataset]:
#     """
#     Partition dataset according to provided indices for each client.
    
#     Parameters
#     ----------
#     dataset: Dataset
#         Full dataset to partition.
#     partition_indices: List[List[int]]
#         List of indices for each client partition.

#     Returns
#     -------
#     subsets: List[Dataset]
#         Subsets of datasets partitioned by client.
#     """
#     subsets = [Subset(dataset, indices) for indices in partition_indices]
#     return subsets



# pylint: disable=too-many-locals
def train_valid_test_partition(
    partitioned_dataset: List[Dataset],
    train_split: float = 0.9,
    validation_split: float = 0.0,
    test_split: float = 0.1,
    random_seed: int = None,
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).

    Parameters
    ----------
    partitioned_dataset: List[Dataset]
        partitioned datasets
    train_split: float
        part of the data used for training
    validation_split: float
        part of the data used for validation
    test_split: float
        part of the data used for testing
    random_seed: int
        seed for data splitting

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    train_subsets = []
    validation_subsets = []
    test_subsets = []

    for subset in partitioned_dataset:
        subset_len = len(subset)
        train_len = int(train_split * subset_len)
        # Do this checkup for full dataset use
        # Consider the case sample size == 5 and
        # train_split = 0.5 test_split = 0.5
        # if such check as below is not performed
        # one sample will be missing
        if validation_split == 0.0:
            test_len = subset_len - train_len
            val_len = 0
        else:
            test_len = int(test_split * subset_len)
            val_len = subset_len - train_len - test_len
        train_dataset, validation_dataset, test_dataset = random_split(
            subset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_subsets.append(train_dataset)
        validation_subsets.append(validation_dataset)
        test_subsets.append(test_dataset)
    return train_subsets, validation_subsets, test_subsets


def transform_datasets_into_dataloaders(
    datasets: List[Dataset], **dataloader_kwargs
) -> List[DataLoader]:
    """Transform datasets into dataloaders.

    Parameters
    ----------
    datasets: List[Dataset]
        list of datasets
    dataloader_kwargs
        arguments to DataLoader

    Returns
    -------
    dataloader: List[DataLoader]
        list of dataloaders
    """
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset, **dataloader_kwargs))
    return dataloaders


def dirichlet_partition(
    df_info: pd.DataFrame, n_clients: int, alpha: float, random_seed: int
) -> List[List[int]]:
    """
    Partition the dataset using a Dirichlet distribution.

    Parameters
    ----------
    df_info: pd.DataFrame
        DataFrame containing writer_id information.
    n_clients: int
        Number of clients.
    alpha: float
        Dirichlet distribution parameter. Lower values make partitions more uneven.
    random_seed: int
        Random seed for reproducibility.

    Returns
    -------
    partition_indices: List[List[int]]
        List of lists where each inner list contains indices for one client's partition.
    """
    np.random.seed(random_seed)
    writer_ids = df_info["writer_id"].values
    unique_writers = np.unique(writer_ids)
    indices = {
        writer_id: np.where(writer_ids == writer_id)[0].tolist()
        for writer_id in unique_writers
    }
    
    # Create a list of indices for each client
    partition_indices = [[] for _ in range(n_clients)]
    
    # Iterate through each writer and distribute samples according to the Dirichlet distribution
    for writer_id, writer_indices in indices.items():
        # Sample proportions for each client from a Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * n_clients)
        
        # Shuffle the writer's indices
        np.random.shuffle(writer_indices)
        
        # Determine the number of samples to assign to each client
        client_sample_counts = (proportions * len(writer_indices)).astype(int)
        
        # Adjust sample counts to ensure all samples are used
        client_sample_counts[-1] += len(writer_indices) - sum(client_sample_counts)
        
        # Split writer indices according to the counts and add to each client's partition
        start_idx = 0
        for client_id in range(n_clients):
            end_idx = start_idx + client_sample_counts[client_id]
            partition_indices[client_id].extend(writer_indices[start_idx:end_idx])
            start_idx = end_idx

    # Prepare to save the results to a file
    output_dir = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "data_distribution_by_client.txt")

    # Print each client's character distribution and number of samples
    with open(output_file, "w") as file:
        for client_id, client_indices in enumerate(partition_indices):
            client_characters = df_info.iloc[client_indices]["character"].fillna('0').values
            character_counts = pd.Series(client_characters).value_counts()
            sorted_counts = character_counts.sort_index(key=lambda x: [ord(c) for c in x])
            sorted_counts_str = ", ".join([f"({char}: {count})" for char, count in sorted_counts.items()])
            file.write(f"Client {client_id}:\n")
            file.write(f"{sorted_counts_str}\n")
            file.write(f"Number of samples: {len(client_indices)}\n\n")
    
    return partition_indices

def validate_proportion(character, proportion, samples_per_client, indices_by_character):
    character_indices = indices_by_character[character]
    num_samples = int(proportion * samples_per_client)
    return num_samples <= len(character_indices)


def dirichlet_partition_by_character(
    df_info: pd.DataFrame, n_clients: int, alpha: float, random_seed: int
) -> List[List[int]]:
    """
    Partition the dataset using a Dirichlet distribution by character class,
    ensuring each client has samples from all character classes.

    Parameters
    ----------
    df_info: pd.DataFrame
        DataFrame containing character information.
    n_clients: int
        Number of clients.
    alpha: float
        Dirichlet distribution parameter. Lower values make partitions more uneven.
    random_seed: int
        Random seed for reproducibility.

    Returns
    -------
    partition_indices: List[List[int]]
        List of lists where each inner list contains indices for one client's partition.
    """
    np.random.seed(random_seed)
    characters = df_info["character"].values
    unique_characters = np.unique(characters)
    
    # Group indices of samples by character
    indices_by_character = {
        character: np.where(characters == character)[0].tolist()
        for character in unique_characters
    }

    # Calculate and print the number of samples for each character and the total
    total_items = 0
    min_samples = float('inf')
    max_samples = float('-inf')
    min_character = None
    max_character = None

    # Prepare to save the results to a file
    output_dir = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "min_max_item_per_class.txt")

    with open(output_file, "a") as file:
        for character, indices in indices_by_character.items():
            num_items = len(indices)
            total_items += num_items
            print(f"Character: {character}, Number of items: {num_items}")
            file.write(f"Character: {character}, Number of items: {num_items}\n")

            # Update min and max samples
            if num_items < min_samples:
                min_samples = num_items
                min_character = character
            if num_items > max_samples:
                max_samples = num_items
                max_character = character
        


        file.write(f"Total number of items across all characters: {total_items}\n")
        file.write(f"Minimum number of items for a character: {min_samples} (Character: {min_character})\n")
        file.write(f"Maximum number of items for a character: {max_samples} (Character: {max_character})\n")
    
    # Create a list of empty lists for each client
    partition_indices = [[] for _ in range(n_clients)]
    
    samples_per_client = 10000
    # samples_per_client = total_items/n_clients

    # Open the file to save the proportions
    with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/proportions.txt", "w") as file:
        # Iterate through each client and assign samples using the Dirichlet distribution
        for client_id in range(n_clients):
            valid_proportions = False
            max_iterations = 10000  # Set a maximum number of iterations to avoid dead loop
            iteration_count = 0
            
            while not valid_proportions and iteration_count < max_iterations:
                iteration_count += 1
                
                # Sample proportions for this client from a Dirichlet distribution
                proportions = np.random.dirichlet(np.repeat(alpha, len(unique_characters)))
                # proportions = np.random.dirichlet([alpha] * len(unique_characters))
                
                # Check if the proportions are valid using multi-threading
                valid_proportions = True
                with ThreadPoolExecutor(max_workers=128) as executor:  # Specify the number of threads
                    futures = [
                        executor.submit(validate_proportion, character, proportions[i], samples_per_client, indices_by_character)
                        for i, character in enumerate(unique_characters)
                    ]
                    for future in as_completed(futures):
                        if not future.result():
                            valid_proportions = False
                            break
                
                # If proportions are not valid, resample
                if not valid_proportions:
                    continue
            
            if not valid_proportions:
                raise RuntimeError(f"Failed to find valid proportions for client {client_id} after {max_iterations} iterations")
            
            
            # Print and save the proportions for each unique character
            for char, proportion in zip(unique_characters, proportions):
                print(f"Client {client_id}, Character {char}: {proportion}")
                file.write(f"Client {client_id}, Character {char}: {proportion}\n")
            
            # Iterate through each character to distribute samples according to the proportions
            for i, character in enumerate(unique_characters):
                character_indices = indices_by_character[character]
                
                # Shuffle the character's indices to randomize the order
                np.random.shuffle(character_indices)
                
                # Calculate the number of samples to assign to this client for this character
                num_samples = int(proportions[i] * samples_per_client)
                
                # Assign the calculated number of samples to this client
                partition_indices[client_id].extend(character_indices[:num_samples])
                
                # Update the indices list for the remaining clients
                # indices_by_character[character] = character_indices[num_samples:]
    
    # Calculate and save the data distribution for each client
    client_distributions = []
    for client_id, indices in enumerate(partition_indices):
        client_chars = df_info.iloc[indices]["character"].values
        char_counts = Counter(client_chars)
        client_distributions.append(f"Client {client_id}: {dict(char_counts)}")

    # Save the distribution information to the specified file
    output_path = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/dirichlet_distribution_by_character.txt"
    with open(output_path, "w") as file:
        file.write("\n".join(client_distributions))
    
    return partition_indices


def dirichlet_partition_by_character_with_even(
    df_info: pd.DataFrame, n_clients: int, alpha: float, random_seed: int
) -> List[List[int]]:
    """
    Partition the dataset using a Dirichlet distribution by character class,
    ensuring each client has samples from all character classes.

    Parameters
    ----------
    df_info: pd.DataFrame
        DataFrame containing character information.
    n_clients: int
        Number of clients.
    alpha: float
        Dirichlet distribution parameter. Lower values make partitions more uneven.
    random_seed: int
        Random seed for reproducibility.

    Returns
    -------
    partition_indices: List[List[int]]
        List of lists where each inner list contains indices for one client's partition.
    """
    np.random.seed(random_seed)
    characters = df_info["character"].values
    unique_characters = np.unique(characters)
    
    # Group indices of samples by character
    indices_by_character = {
        character: np.where(characters == character)[0].tolist()
        for character in unique_characters
    }

    # Calculate and print the number of samples for each character and the total
    total_items = 0
    min_samples = float('inf')
    max_samples = float('-inf')
    min_character = None
    max_character = None

    # Prepare to save the results to a file
    output_dir = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "min_max_item_per_class.txt")

    with open(output_file, "a") as file:
        for character, indices in indices_by_character.items():
            num_items = len(indices)
            total_items += num_items
            print(f"Character: {character}, Number of items: {num_items}")
            file.write(f"Character: {character}, Number of items: {num_items}\n")

            # Update min and max samples
            if num_items < min_samples:
                min_samples = num_items
                min_character = character
            if num_items > max_samples:
                max_samples = num_items
                max_character = character
        
    # print(f"Total number of items across all characters: {total_items}")
    # print(f"Minimum number of items for a character: {min_samples} (Character: {min_character})")
    # print(f"Maximum number of items for a character: {max_samples} (Character: {max_character})")
    # Save the results to a file

        file.write(f"Total number of items across all characters: {total_items}\n")
        file.write(f"Minimum number of items for a character: {min_samples} (Character: {min_character})\n")
        file.write(f"Maximum number of items for a character: {max_samples} (Character: {max_character})\n")
    
    # Create a list of empty lists for each client
    partition_indices = [[] for _ in range(n_clients)]
    
    # samples_per_client = 10000
    samples_per_client = total_items/n_clients

    # Open the file to save the proportions
    with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/proportions.txt", "w") as file:
        # Iterate through each client and assign samples using the Dirichlet distribution
        for client_id in range(n_clients):
            valid_proportions = False
            while not valid_proportions:
                # Sample proportions for this client from a Dirichlet distribution
                # proportions = np.random.dirichlet(np.repeat(alpha, len(unique_characters)))
                # proportions = np.random.dirichlet([alpha] * len(unique_characters))
                #=============================================================================
                proportions = np.full(len(unique_characters), 1 / len(unique_characters))

                # Check if the proportions are valid
                valid_proportions = True
                for i, character in enumerate(unique_characters):
                    character_indices = indices_by_character[character]
                    num_samples = int(proportions[i] * samples_per_client)

                    if num_samples > len(character_indices):
                        valid_proportions = False
                        break
            
            # Print and save the proportions for each unique character
            for char, proportion in zip(unique_characters, proportions):
                print(f"Client {client_id}, Character {char}: {proportion}")
                file.write(f"Client {client_id}, Character {char}: {proportion}\n")
            
            # Iterate through each character to distribute samples according to the proportions
            for i, character in enumerate(unique_characters):
                character_indices = indices_by_character[character]
                
                # Shuffle the character's indices to randomize the order
                np.random.shuffle(character_indices)
                
                # Calculate the number of samples to assign to this client for this character
                num_samples = int(proportions[i] * samples_per_client)
                
                # Assign the calculated number of samples to this client
                partition_indices[client_id].extend(character_indices[:num_samples])
                
                # Update the indices list for the remaining clients
                # indices_by_character[character] = character_indices[num_samples:]
    
    # Calculate and save the data distribution for each client
    client_distributions = []
    for client_id, indices in enumerate(partition_indices):
        client_chars = df_info.iloc[indices]["character"].values
        char_counts = Counter(client_chars)
        client_distributions.append(f"Client {client_id}: {dict(char_counts)}")

    # Save the distribution information to the specified file
    output_path = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/dirichlet_distribution_by_character.txt"
    with open(output_path, "w") as file:
        file.write("\n".join(client_distributions))
    
    return partition_indices

def dirichlet_partition_by_character_with_overlap(
    df_info: pd.DataFrame, n_clients: int, alpha: float, random_seed: int
) -> List[List[int]]:
    """
    Partition the dataset using a Dirichlet distribution by character class,
    ensuring each client has samples from all character classes.

    Parameters
    ----------
    df_info: pd.DataFrame
        DataFrame containing character information.
    n_clients: int
        Number of clients.
    alpha: float
        Dirichlet distribution parameter. Lower values make partitions more uneven.
    random_seed: int
        Random seed for reproducibility.

    Returns
    -------
    partition_indices: List[List[int]]
        List of lists where each inner list contains indices for one client's partition.
    """
    np.random.seed(random_seed)
    characters = df_info["character"].values
    unique_characters = np.unique(characters)
    
    # Group indices of samples by character
    indices_by_character = {
        character: np.where(characters == character)[0].tolist()
        for character in unique_characters
    }

    # Calculate and print the number of samples for each character and the total
    total_items = 0
    min_samples = float('inf')
    max_samples = float('-inf')
    min_character = None
    max_character = None

    # Prepare to save the results to a file
    output_dir = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "min_max_item_per_class.txt")

    with open(output_file, "a") as file:
        for character, indices in indices_by_character.items():
            num_items = len(indices)
            total_items += num_items
            print(f"Character: {character}, Number of items: {num_items}")
            file.write(f"Character: {character}, Number of items: {num_items}\n")

            # Update min and max samples
            if num_items < min_samples:
                min_samples = num_items
                min_character = character
            if num_items > max_samples:
                max_samples = num_items
                max_character = character
        


        file.write(f"Total number of items across all characters: {total_items}\n")
        file.write(f"Minimum number of items for a character: {min_samples} (Character: {min_character})\n")
        file.write(f"Maximum number of items for a character: {max_samples} (Character: {max_character})\n")
    
    # Create a list of empty lists for each client
    partition_indices = [[] for _ in range(n_clients)]
    
    samples_per_client = 10000
    # samples_per_client = total_items/n_clients

    # Open the file to save the proportions
    with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/proportions.txt", "w") as file:
        # Iterate through each client and assign samples using the Dirichlet distribution
        for client_id in range(n_clients):
            valid_proportions = False
            max_iterations = 10000  # Set a maximum number of iterations to avoid dead loop
            iteration_count = 0
            
            while not valid_proportions and iteration_count < max_iterations:
                iteration_count += 1
                
                # Sample proportions for this client from a Dirichlet distribution
                proportions = np.random.dirichlet(np.repeat(alpha, len(unique_characters)))
                # proportions = np.random.dirichlet([alpha] * len(unique_characters))
                
                # Check if the proportions are valid using multi-threading
                valid_proportions = True
                with ThreadPoolExecutor(max_workers=128) as executor:  # Specify the number of threads
                    futures = [
                        executor.submit(validate_proportion, character, proportions[i], samples_per_client, indices_by_character)
                        for i, character in enumerate(unique_characters)
                    ]
                    for future in as_completed(futures):
                        if not future.result():
                            valid_proportions = False
                            break
                
                # If proportions are not valid, resample
                if not valid_proportions:
                    continue
            
            if not valid_proportions:
                raise RuntimeError(f"Failed to find valid proportions for client {client_id} after {max_iterations} iterations")
            
            # Print and save the proportions for each unique character
            for char, proportion in zip(unique_characters, proportions):
                print(f"Client {client_id}, Character {char}: {proportion}")
                file.write(f"Client {client_id}, Character {char}: {proportion}\n")
            
            # Iterate through each character to distribute samples according to the proportions
            for i, character in enumerate(unique_characters):
                character_indices = indices_by_character[character]
                
                # Shuffle the character's indices to randomize the order
                np.random.shuffle(character_indices)
                
                # Calculate the number of samples to assign to this client for this character
                num_samples = int(proportions[i] * samples_per_client)
                
                # Assign the calculated number of samples to this client
                partition_indices[client_id].extend(character_indices[:num_samples])
                
                # Update the indices list for the remaining clients
                # indices_by_character[character] = character_indices[num_samples:]
    
    # Calculate and save the data distribution for each client
    client_distributions = []
    for client_id, indices in enumerate(partition_indices):
        client_chars = df_info.iloc[indices]["character"].values
        char_counts = Counter(client_chars)
        client_distributions.append(f"Client {client_id}: {dict(char_counts)}")

    # Save the distribution information to the specified file
    output_path = "baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/dirichlet_distribution_by_character.txt"
    with open(output_path, "w") as file:
        file.write("\n".join(client_distributions))

    partition_indices = apply_overlap(partition_indices, overlap_percent=0.25,overlap_clients=20)
    
    return partition_indices


def apply_overlap(partition_indices, overlap_percent, overlap_clients=0):
    """
    Apply dynamic overlap logic to the partitioned indices for a specified number of clients.

    Parameters
    ----------
    partition_indices: List[List[int]]
        List of lists where each inner list contains indices for one client's partition.
    df_info: pd.DataFrame
        DataFrame containing character information.
    overlap_percent: float
        Percentage of overlap between clients.
    overlap_clients: int
        Number of clients that will share overlapping indices. If 0, no overlap is applied.

    Returns
    -------
    partition_indices: List[List[int]]
        Modified partition indices with overlap applied.
    """
    if overlap_clients == 0:
        print("No overlap applied as overlap_clients is set to 0.")
        return partition_indices

    # list of overlap clients
    overlap_client_ids = list(range(overlap_clients))

    # Initialize overlap logic
    overlap_seed = 55
    rng_overlap = np.random.default_rng(overlap_seed)

    # Calculate the number of samples to copy for each overlap client
    num_samples_to_copy = [
        int(len(partition_indices[client]) * overlap_percent)
        for client in overlap_client_ids
    ]

    # create set for overlapping client
    copied_indices = {client: set() for client in overlap_client_ids}

    # Loop over contributing clients (clients not in overlap_client_ids)
    for client_id in range(len(partition_indices)):
        if client_id not in overlap_client_ids:
            available_indices = list(
                set(partition_indices[client_id]) - set.union(*copied_indices.values())
            )

            # Distribute indices to overlap clients
            for i, client in enumerate(overlap_client_ids):
                num_samples_per_contributing_client = num_samples_to_copy[i] // (
                    len(partition_indices) - len(overlap_client_ids)
                )

                # Ensure there are enough available indices
                assert len(available_indices) >= num_samples_per_contributing_client, \
                    f"Not enough available indices for client {client_id} to contribute to client {client}"
                
                selected_indices = set(
                    available_indices[:num_samples_per_contributing_client]
                ) #got the first num_samples_per_contributing_client from available_indices
                copied_indices[client].update(selected_indices)
                available_indices = available_indices[
                    num_samples_per_contributing_client:
                ]
                # print(f"Client {client_id} contribute Overlap_Client: {client}\n Selected_Samples: {selected_indices}\n Available: {available_indices}")

                # Print details for verification

                with open("baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist/plot/overlap_client_verify.txt", "a") as file:
                    file.write(f"Client {client_id} contributes {len(selected_indices)} samples to Client {client}\n")
                    file.write(f"Selected indices: {selected_indices}\n")
                    file.write(f"Remaining available indices for Client {client_id}: {available_indices}\n")

    # Shuffle and assign copied indices back to overlap clients
    for client in overlap_client_ids:
        copied_indices_list = list(copied_indices[client])
        # rng_overlap.shuffle(copied_indices_list)
        partition_indices[client][: len(copied_indices_list)] = copied_indices_list

        overlap_percent_actual = (
            len(copied_indices_list) / len(partition_indices[client]) * 100
        )
        print(f"Client {client} has {len(copied_indices_list)} overlapping samples, which is {overlap_percent_actual:.2f}% of their total samples.")

    return partition_indices

def update_class_count(class_count, indices, all_labels):
    for idx in indices:
        class_count[all_labels[idx]] += 1
    return class_count

# pylint: disable=too-many-arguments
def create_federated_dataloaders(
    sampling_type: str,
    dataset_fraction: float,
    batch_size: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """Create the federated dataloaders by following all the preprocessing
    steps and division.

    Parameters
    ----------
    sampling_type: str
        "niid" or "iid"
    dataset_fraction: float
        fraction of the total data that will be used for sampling
    batch_size: int
        batch size
    train_fraction, validation_fraction, test_fraction: float
        fraction of each local dataset used for training, validation, testing
    random_seed: int
        random seed for data shuffling

    Returns
    -------
    """
    if train_fraction + validation_fraction + test_fraction != 1.0:
        raise ValueError(
            "The fraction of train, validation and test should add up to 1.0."
        )
    # Download and unzip the data
    log(INFO, "NIST data downloading started")
    nist_by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    nist_by_writer_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    nist_by_class_downloader = ZipDownloader("by_class", "data/raw", nist_by_class_url)
    nist_by_writer_downloader = ZipDownloader(
        "by_write", "data/raw", nist_by_writer_url
    )
    nist_by_class_downloader.download()
    nist_by_writer_downloader.download()
    log(INFO, "NIST data downloading done")

    # Preprocess the data
    log(INFO, "Preprocessing of the NIST data started")
    nist_data_path = pathlib.Path("data")
    nist_preprocessor = NISTPreprocessor(nist_data_path)
    nist_preprocessor.preprocess()
    log(INFO, "Preprocessing of the NIST data done")

    # Create information for sampling
    log(INFO, "Creation of the sampling information started")
    df_info_path = pathlib.Path("data/processed_FeMNIST/processed_images_to_labels.csv")
    df_info = pd.read_csv(df_info_path, index_col=0)
    sampler = NistSampler(df_info)
    sampled_data_info = sampler.sample(
        sampling_type, dataset_fraction, random_seed=random_seed
    )
    sampled_data_info_path = pathlib.Path(
        f"data/processed_FeMNIST/{sampling_type}_sampled_images_to_labels.csv"
    )
    sampled_data_info.to_csv(sampled_data_info_path)
    log(INFO, "Creation of the sampling information done")

    # Create a list of DataLoaders
    log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets started")
    sampled_data_info = pd.read_csv(sampled_data_info_path)
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(sampled_data_info["character"])

#==============================================================================================================
    # full_dataset = create_dataset(sampled_data_info, labels)
    # division_list = create_partition_list(sampled_data_info)
    # partitioned_dataset = partition_dataset(full_dataset, division_list)
#==============================================================================================================
    
    full_dataset = create_dataset(sampled_data_info, labels)
    if sampling_type == "niid":
        # #dirichlet distribution by client
        partition_indices = dirichlet_partition(sampled_data_info, n_clients=10, alpha=0.9, random_seed=random_seed)

        #dirichlet distribution by class
        # partition_indices = dirichlet_partition_by_character( sampled_data_info, n_clients=100, alpha=0.9, random_seed=random_seed)
        # partition_indices = dirichlet_partition_by_character_with_even( sampled_data_info, n_clients=100, alpha=0.9, random_seed=random_seed)
        # partition_indices = dirichlet_partition_by_character_with_overlap( sampled_data_info, n_clients=100, alpha=0.9, random_seed=random_seed)
    else:
        raise ValueError("Only 'niid' sampling is supported with Dirichlet partitioning.")
    partitioned_dataset = partition_dataset(full_dataset, partition_indices)

    (
        partitioned_train,
        partitioned_validation,
        partitioned_test,
    ) = train_valid_test_partition(
        partitioned_dataset,
        random_seed=random_seed,
        train_split=train_fraction,
        validation_split=validation_fraction,
        test_split=test_fraction,
    )
    trainloaders = transform_datasets_into_dataloaders(
        partitioned_train, batch_size=batch_size
    )
    valloaders = transform_datasets_into_dataloaders(
        partitioned_validation, batch_size=batch_size
    )
    testloaders = transform_datasets_into_dataloaders(
        partitioned_test, batch_size=batch_size
    )
    log(INFO, "Creation of the partitioned by writer_id PyTorch Datasets done")
    return trainloaders, valloaders, testloaders
