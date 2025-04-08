from typing import Any

import hydra
import pytorch_lightning as pl
import torch

from torchmetrics import *
import transformers
from transformers import Adafactor

from maverick.common.metrics import *
from maverick.common.util import *
from maverick.models.model_incr import *
from maverick.models.model_mes import Maverick_mes
from maverick.models.model_s2e import *

def build_mention_maps(clusters):
    mention_to_cluster = {}
    for cluster in clusters:
        cluster_tuple = tuple(sorted(cluster)) # Use a canonical representation
        for mention in cluster:
            mention_to_cluster[mention] = cluster_tuple
    return mention_to_cluster

def filter_clusters_by_mentions(
    clusters, mentions_to_keep_set):
    """
    Filters clusters to only include mentions present in mentions_to_keep_set.
    Removes empty clusters. Resulting clusters are tuples.
    """
    filtered_clusters = []
    for cluster in clusters:
        # Filter mentions within the cluster
        new_cluster_mentions = [
            mention for mention in cluster if mention in mentions_to_keep_set
        ]
        # Only add if the filtered cluster is not empty
        if new_cluster_mentions:
            # Convert back to tuple for consistency
            filtered_clusters.append(tuple(sorted(new_cluster_mentions)))
    return filtered_clusters

def filter_mention_map(mention_map, mentions_to_keep_set):
    """Filters a mention-to-cluster map based on a set of mentions to keep."""
    filtered_map = {}
    for mention, cluster in mention_map.items():
        if mention in mentions_to_keep_set:
            # Also filter the cluster tuple itself to only contain kept mentions
            filtered_cluster = tuple(sorted([
                m for m in cluster if m in mentions_to_keep_set
            ]))
            # Ensure the filtered cluster is not empty before adding
            if filtered_cluster:
                filtered_map[mention] = filtered_cluster
    return filtered_map


class BasePLModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        try:
            self.model = hydra.utils.instantiate(self.hparams.model)
        except:
            self.hparams.model["_target_"] = "maverick." + self.hparams.model["_target_"]
            self.model = hydra.utils.instantiate(self.hparams.model)
        self.train_step_predictions = []
        self.train_step_gold = []
        self.validation_step_predictions = []
        self.validation_step_gold = []
        self.test_step_predictions = []
        self.test_step_gold = []

    def forward(self, batch) -> dict:
        output_dict = self.model(batch, "forward")
        return output_dict

    def evaluate(self, predictions, golds):
        mention_evaluator = OfficialMentionEvaluator()
        start_evaluator = OfficialMentionEvaluator()
        cluster_mention_evaluator = OfficialMentionEvaluator()
        coref_evaluator = OfficialCoNLL2012CorefEvaluator()
        singleton_evaluator = OfficialSingletonEvaluator()
        kg_subset_conll_evaluator = KGSubsetCoNLL2012CorefEvaluator()
        result = {}

        for pred, gold in zip(predictions, golds):
            # Evaluate singletons:
            if "gold_singletons" in pred and "singletons" in pred:
                # Convert to lists if the values are tensors.
                pred_singletons = pred["singletons"]
                gold_singletons = pred["gold_singletons"]
                if torch.is_tensor(pred_singletons):
                    pred_singletons = pred_singletons.cpu().tolist()
                if torch.is_tensor(gold_singletons):
                    gold_singletons = gold_singletons.cpu().tolist()
                # Print converted values.
                #print("Converted pred_singletons:", pred_singletons)
                #print("Converted gold_singletons:", gold_singletons)    
                singleton_evaluator.update(pred_singletons, gold_singletons)

            if "start_idxs" in pred.keys():
                starts_pred = pred["start_idxs"][0].tolist()
                starts_gold = (gold["gold_starts"][0] == 1).nonzero(as_tuple=False).squeeze(-1).tolist()
                start_evaluator.update(starts_pred, starts_gold)

            if "mention_idxs" in pred.keys():
                mentions_pred = [tuple(p) for p in pred["mention_idxs"][0].tolist()]
                mentions_gold = [tuple(g) for g in (gold["gold_mentions"][0] == 1).nonzero(as_tuple=False).tolist()]
                mention_evaluator.update(mentions_pred, mentions_gold)

            if "clusters" in pred.keys():
                pred_clusters = pred["clusters"]
                gold_clusters = gold["gold_clusters"]
                mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
                mention_to_predicted_clusters = extract_mentions_to_clusters(pred_clusters)

                coref_evaluator.update(
                    pred_clusters,
                    gold_clusters,
                    mention_to_predicted_clusters,
                    mention_to_gold_clusters,
                )

                cluster_mention_evaluator.update(
                    [item for sublist in gold_clusters for item in sublist],
                    [item for sublist in pred_clusters for item in sublist],
                )
            # --- KG Subset Coreference Evaluation ---
            kg_enhanced_mentions = pred["kg_enhanced_mentions"]
            mention_to_predicted = build_mention_maps(pred_clusters)
            mention_to_gold = build_mention_maps(gold_clusters)
            if kg_enhanced_mentions and (pred_clusters or gold_clusters):
                filtered_pred_clusters = filter_clusters_by_mentions(pred_clusters, kg_enhanced_mentions)
                filtered_gold_clusters = filter_clusters_by_mentions(gold_clusters, kg_enhanced_mentions)

                filtered_mention_to_predicted = filter_mention_map(mention_to_predicted, kg_enhanced_mentions)
                filtered_mention_to_gold = filter_mention_map(mention_to_gold, kg_enhanced_mentions)

                kg_subset_conll_evaluator.update(
                    filtered_pred_clusters,
                    filtered_gold_clusters,
                    filtered_mention_to_predicted,
                    filtered_mention_to_gold
                )

        p, r, f1 = start_evaluator.get_prf()
        result.update(
            {
                "start_f1_score": f1,
                "start_precision": p,
                "start_recall": r,
            }
        )
        p, r, f1 = mention_evaluator.get_prf()
        result.update({"mention_f1_score": f1, "mention_precision": p, "mention_recall": r})
        p, r, f1 = cluster_mention_evaluator.get_prf()
        result.update(
            {
                "cluster_mention_f1_score": f1,
                "cluster_mention_precision": p,
                "cluster_mention_recall": r,
            }
        )

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = coref_evaluator.get_prf(metric)
            result.update(
                {
                    metric + "_f1_score": f1,
                    metric + "_precision": p,
                    metric + "_recall": r,
                }
            )
        p, r, f1 = singleton_evaluator.get_prf()
        result.update({
        "singleton_f1_score": f1,
        "singleton_precision": p,
        "singleton_recall": r,
        })
        

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = kg_subset_conll_evaluator.get_prf(metric)
            result.update({
                f"kg_subset_{metric}_f1_score": f1, 
                f"kg_subset_{metric}_precision": p, 
                f"kg_subset_{metric}_recall": r})
        return result

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self.model(
            stage="train",
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_mask=batch["eos_mask"],
            raw_gold_clusters=batch["raw_gold_clusters"],
            gold_starts=batch["gold_starts"],
            gold_mentions=batch["gold_mentions"],
            gold_clusters=batch["gold_clusters"],
            tokens=batch["tokens"],
            subtoken_map=batch["subtoken_map"],
            new_token_map=batch["new_token_map"],
            singletons=batch["singletons"],
            step=self.global_step,
        )
        self.log_dict({"train/" + k: v for k, v in output["loss_dict"].items()}, on_step=True)
        return output["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        output = self.model(
            stage="val",
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_mask=batch["eos_mask"],
            raw_gold_clusters=batch["raw_gold_clusters"],
            gold_starts=batch["gold_starts"],
            gold_mentions=batch["gold_mentions"],
            gold_clusters=batch["gold_clusters"],
            tokens=batch["tokens"],
            subtoken_map=batch["subtoken_map"],
            new_token_map=batch["new_token_map"],
            singletons=batch["singletons"],
            step=self.global_step,
        )
        self.log_dict({"val/" + k: v for k, v in output["loss_dict"].items()})
        output["pred_dict"]["clusters"] = original_token_offsets(
            clusters=output["pred_dict"]["clusters"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )
        self.validation_step_predictions.append(output["pred_dict"])
        self.validation_step_gold.append(
            {
                "gold_starts": batch["gold_starts"].cpu(),
                "gold_mentions": batch["gold_mentions"].cpu(),
                "gold_clusters": original_token_offsets(
                    clusters=unpad_gold_clusters(batch["gold_clusters"].cpu()),
                    subtoken_map=batch["subtoken_map"][0],
                    new_token_map=batch["new_token_map"][0],
                ),
            }
        )

    def on_validation_epoch_end(self):
        self.log_dict(
            {"val/" + k: v for k, v in self.evaluate(self.validation_step_predictions, self.validation_step_gold).items()}
        )
        self.validation_step_predictions = []
        self.validation_step_gold = []

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        output = self.model(
            stage="test",
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_mask=batch["eos_mask"],
            raw_gold_clusters=batch["raw_gold_clusters"],
            #eos_indices=batch["eos_indices"],
            tokens=batch["tokens"],
            subtoken_map=batch["subtoken_map"],
            new_token_map=batch["new_token_map"],
            singletons=batch["singletons"],
            step=self.global_step,
        )
        self.log_dict({"test/" + k: v for k, v in output["loss_dict"].items()})
        output["pred_dict"]["clusters"] = original_token_offsets(
            clusters=output["pred_dict"]["clusters"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )
        self.test_step_predictions.append(output["pred_dict"])
        self.test_step_gold.append(
            {
                "gold_starts": batch["gold_starts"].cpu(),
                "gold_mentions": batch["gold_mentions"].cpu(),
                "gold_clusters": original_token_offsets(
                    clusters=unpad_gold_clusters(batch["gold_clusters"].cpu()),
                    subtoken_map=batch["subtoken_map"][0],
                    new_token_map=batch["new_token_map"][0],
                ),
            }
        )

    def on_test_epoch_end(self):
        self.log_dict({"test/" + k: v for k, v in self.evaluate(self.test_step_predictions, self.test_step_gold).items()})
        self.test_step_predictions = []
        self.test_step_gold = []

    def configure_optimizers(self):
        if self.hparams.opt == "RAdam":
            opt = hydra.utils.instantiate(self.hparams.RAdam, params=self.parameters())
            return opt
        else:
            return self.custom_opt()

    def custom_opt(self):
        no_decay = ["bias", "LayerNorm.weight"]
        head_params = ["representaion", "classifier"]

        model_decay = [
            p
            for n, p in self.model.named_parameters()
            if not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        model_no_decay = [
            p
            for n, p in self.model.named_parameters()
            if not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]
        head_decay = [
            p
            for n, p in self.model.named_parameters()
            if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        head_no_decay = [
            p for n, p in self.model.named_parameters() if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]

        head_learning_rate = 3e-4
        lr = 2e-5
        wd = 0.01
        optimizer_grouped_parameters = [
            {"params": model_decay, "lr": lr, "weight_decay": wd},
            {"params": model_no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_learning_rate, "weight_decay": wd},
            {"params": head_no_decay, "lr": head_learning_rate, "weight_decay": 0.0},
        ]
        optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.lr_scheduler.num_training_steps * 0.1,
            num_training_steps=self.hparams.lr_scheduler.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
