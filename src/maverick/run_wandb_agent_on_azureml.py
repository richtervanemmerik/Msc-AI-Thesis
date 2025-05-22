import os
from azureml.core import Workspace, Experiment, Environment
from azureml.core import ScriptRunConfig 
from azureml.core.runconfig import PyTorchConfiguration 
# Optional: If using Azure Key Vault for storing the API key (recommended)
# from azureml.core import Keyvault  
COMPUTE_TARGET_NAME = "richter-gpu-a100"      
ENVIRONMENT_NAME = "maverick"                    
AZUREML_EXPERIMENT_NAME = "wandb-sweep-agent-launcher" 

# WandB Configuration
# ** REQUIRED: Replace this with the Sweep ID you get from `wandb sweep sweep.yaml` **
WANDB_SWEEP_ID = "ai4fintech/maverick-coref-maverick/ikkd9py6" 

# Set this environment variable in your terminal before running the script:
# export WANDB_API_KEY='your_key_here'
WANDB_API_KEY = os.getenv("WANDB_API_KEY") 

# Concurrency Configuration
NUM_CONCURRENT_AGENTS = 3  # Set this to the number of runs you want concurrently

# --- Script ---

def main():
    # 1. Connect to Azure ML Workspace
    try:
        # Tries to load from config file first
        ws = Workspace.from_config() 
        print(f"Connected to workspace '{ws.name}' from configuration file.")
    except Exception as e:
        print(f"Error connecting to Azure ML workspace: {e}")
        print("Ensure you are logged in (`az login`), the correct subscription is set (`az account set -s ...`),")
        print("and that your config file (~/.azureml/config.json) exists or you provide explicit details.")
        return

    # --- Get WANDB API Key ---
    api_key_to_use = WANDB_API_KEY 

    # --- Final Checks ---
    if not api_key_to_use:
        print("ERROR: WANDB_API_KEY is not set.")
        print("Please set the WANDB_API_KEY environment variable, hardcode it (not recommended),")
        print("or configure and use Key Vault access.")
        return
        
    if not WANDB_SWEEP_ID or "YOUR_USERNAME" in WANDB_SWEEP_ID:
         print("ERROR: WANDB_SWEEP_ID is not set correctly.")
         print("Please replace 'YOUR_USERNAME/YOUR_PROJECT/YOUR_SWEEP_ID' with your actual Sweep ID.")
         return

    # 2. Get Azure ML Environment
    try:
        environment = Environment.get(workspace=ws, name=ENVIRONMENT_NAME)
        print(f"Found environment: {environment.name} (Version: {environment.version})")
    except Exception as e:
        print(f"Error getting environment '{ENVIRONMENT_NAME}': {e}")
        print(f"Ensure the environment '{ENVIRONMENT_NAME}' is registered in the '{ws.name}' workspace.")
        return

    # 3. Define the Command to Run the WandB Agent
    agent_arguments = [WANDB_SWEEP_ID]
    # The 'script' for ScriptRunConfig will effectively be 'wandb'
    agent_script_name = "run_wandb_agent.sh"

    # 4. Configure the Azure ML Command Job
    dist_config = PyTorchConfiguration(process_count=NUM_CONCURRENT_AGENTS, node_count=1)

    # --- Create ScriptRunConfig ---
    src = ScriptRunConfig(
        source_directory=".", 
        script=agent_script_name,
        arguments=agent_arguments,
        compute_target=COMPUTE_TARGET_NAME,
        environment=environment, # Use the fetched environment object
        distributed_job_config=dist_config 
    )
    # --- Set environment variables on the RunConfiguration ---
    src.run_config.environment_variables = {
            "WANDB_API_KEY": api_key_to_use 
        }

    # 5. Submit the Job to an Azure ML Experiment
    experiment = Experiment(workspace=ws, name=AZUREML_EXPERIMENT_NAME)
    print(f"\nSubmitting job using ScriptRunConfig to start {NUM_CONCURRENT_AGENTS} WandB agent(s)...")
    run = experiment.submit(config=src) # Submit the ScriptRunConfig

    # 6. Provide Feedback to the User
    print(f"\nAzure ML Run Submitted!")
    print(f"Run ID: {run.id}")
    # Provide a direct link to the Azure ML run details page
    print(f"View Azure ML Run Details: {run.get_portal_url()}")
    # Construct the likely WandB sweep URL for easy access
    wandb_sweep_url_base = WANDB_SWEEP_ID.replace('/','/',1).replace('/','/sweeps/',1) 
    print(f"Monitor WandB Sweep Progress: https://wandb.ai/{wandb_sweep_url_base}")
    print(f"\nAgents running in Azure ML will pick up jobs from sweep '{WANDB_SWEEP_ID}'.")
    print("You can terminate the agents by canceling the Azure ML Run.")


if __name__ == "__main__":
    main()