# azure_submit.py  – launch from your workstation or CI
import json
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import AzureCliCredential

# Docker image & Azure ML compute
BASE_IMAGE = (
    "kpmgnldankfwepacr.azurecr.io/richter-master-thesis/maverick-training:latest"
)
ENVIRONMENT_NAME = "maverick"
COMPUTE_NAME = "richter-gpu-a100-sw"
CONDA_FILE_PATH = (
    "/home/azureuser/cloudfiles/code/Users/rvanemmerik1/Msc-AI-Thesis/maverick.yml"
)


COMMAND = (
    "python train.py "
    # -------- optimiser -------------------------------------
    "module.Adafactor.lr=${{inputs.learning_rate}} "
    "module.Adafactor.weight_decay=${{inputs.weight_decay}} "
    # -------- scheduler -------------------------------------
    "module.lr_scheduler.num_warmup_steps=${{inputs.warmup_steps}} "
    "module.lr_scheduler.num_training_steps=${{inputs.training_steps}} "
    # -------- model knobs ------------------------------------
    "train.model_name=${{inputs.model_name}} "                        
    "module.model.huggingface_model_name=answerdotai/${{inputs.model_name}} "
    "module.model.incremental_model_num_layers=${{inputs.incremental_layers}} "
    "module.model.kg_fusion_strategy=${{inputs.kg_fusion_strategy}} "
    "module.model.kg_unknown_handling=${{inputs.kg_unknown_handling}} "
    "module.model.use_random_kg_all=${{inputs.use_random_kg_all}} "
    "module.model.use_random_kg_selective=${{inputs.use_random_kg_selective}} "
    # -------- trainer ---------------------------------------
    "pl_trainer.accumulate_grad_batches=${{inputs.gradient_accumulation_steps}} "
    "pl_trainer.max_epochs=${{inputs.epochs}} "
    "pl_trainer.early_stopping_callback.patience=${{inputs.patience}} "
    # -------- data ------------------------------------------
    "defaults.data=${{inputs.data}} "
)

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
if __name__ == "__main__":
    cfg = json.load(open("config.json"))
    ml_client = MLClient(
        AzureCliCredential(),
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )

    # build / reuse the docker environment
    ml_client.environments.create_or_update(
        Environment(name=ENVIRONMENT_NAME, image=BASE_IMAGE)
    )

    job_input_args = dict(
        model_name="answerdotai/ModernBERT-base",
        epochs=50,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=500,
        training_steps=5000,
        incremental_layers=1,
        kg_fusion_strategy="baseline",
        kg_unknown_handling="unk_embed",
        use_random_kg_all=False,
        use_random_kg_selective=False,
        gradient_accumulation_steps=2,
        patience=120,
    )

    ml_client.create_or_update(
        command(
            code="./maverick/",
            command=COMMAND,
            inputs=job_input_args,
            environment=f"{ENVIRONMENT_NAME}@latest",
            compute=COMPUTE_NAME,
        )
    )
