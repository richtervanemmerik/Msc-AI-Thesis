import json
import random

def sample_random_documents(input_file, output_file, sample_size):
    """
    Randomly sample a specified number of documents from a JSONL file.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output JSONL file where the sampled documents will be saved.
        sample_size (int): Number of documents to sample.
    """
    # Read all documents from the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as infile:
        documents = [json.loads(line) for line in infile]

    # Ensure the sample size does not exceed the number of available documents
    if sample_size > len(documents):
        raise ValueError(f"Sample size ({sample_size}) exceeds the number of available documents ({len(documents)}).")

    # Randomly sample the specified number of documents
    sampled_documents = random.sample(documents, sample_size)

    # Write the sampled documents to the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for doc in sampled_documents:
            outfile.write(json.dumps(doc) + '\n')

# Example usage
input_train_file = '/home/azureuser/cloudfiles/code/Users/rvanemmerik1/Msc-AI-Thesis/maverick/data/prepare_preco/train_36120_preprocessed_preco.jsonl'  # Path to the train PreCo dataset
output_sample_file = '/home/azureuser/cloudfiles/code/Users/rvanemmerik1/Msc-AI-Thesis/maverick/data/prepare_preco/train_600_preprocessed_preco.jsonl'  # Path to save the sampled documents
sample_size = 600  # Number of documents to sample

sample_random_documents(input_train_file, output_sample_file, sample_size)