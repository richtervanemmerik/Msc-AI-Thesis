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
COMPUTE_NAME = "richter-se-a100"
CONDA_FILE_PATH = (
    "/home/azureuser/cloudfiles/code/Users/rvanemmerik1/Msc-AI-Thesis/maverick.yml"
)
EXPERIMENT_NAME = "complex-preco"

COMMAND = (
    "python train.py "
    # -------- optimiser -------------------------------------
    "model.module.Adafactor.lr=${{inputs.learning_rate}} "
    "model.module.Adafactor.weight_decay=${{inputs.weight_decay}} "
    # -------- scheduler -------------------------------------
    "model.module.lr_scheduler.num_warmup_steps=${{inputs.warmup_steps}} "
    "model.module.lr_scheduler.num_training_steps=${{inputs.training_steps}} "
    # -------- model knobs ------------------------------------
    "train.model_name=${{inputs.model_name}} "                        
    "model.module.model.huggingface_model_name=${{inputs.model_name}} "
    "model.module.model.incremental_model_num_layers=${{inputs.incremental_layers}} "
    "model.module.model.kg_fusion_strategy=${{inputs.kg_fusion_strategy}} "
    "model.module.model.kg_unknown_handling=${{inputs.kg_unknown_handling}} "
    "model.module.model.use_random_kg_all=${{inputs.use_random_kg_all}} "
    "model.module.model.use_random_kg_selective=${{inputs.use_random_kg_selective}} "
    "model.module.model.kge_model_name=${{inputs.kge_model_name}} "
    # -------- trainer ---------------------------------------
    "train.pl_trainer.accumulate_grad_batches=${{inputs.gradient_accumulation_steps}} "
    "train.pl_trainer.max_epochs=${{inputs.epochs}} "
    "train.early_stopping_callback.patience=${{inputs.patience}} "
    # -------- data ------------------------------------------
    "data=${{inputs.data}} "
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
        warmup_steps=1500,
        training_steps=15000,
        incremental_layers=1,
        kg_fusion_strategy="fusion",
        kg_unknown_handling="unk_embed",
        use_random_kg_all=False,
        use_random_kg_selective=False,
        gradient_accumulation_steps=2,
        patience=120,
        data="preco",
        kge_model_name="TransE",
    )

    ml_client.create_or_update(
        command(
            code="./src/",
            command=COMMAND,
            inputs=job_input_args,
            environment=f"{ENVIRONMENT_NAME}@latest",
            compute=COMPUTE_NAME,
            environment_variables={
                "WANDB_API_KEY": "ed6f1e0fdb4796d6a97528308f1bbc0aa4c043fe",
                "HYDRA_FULL_ERROR": 1,
                "HF_HOME": "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gsidiropoulos2/code/Users/rvanemmerik1/jobs-outputs/{EXPERIMENT_NAME}/hf_cache",
            },
        )
    )
