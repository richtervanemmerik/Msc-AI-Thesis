import torch
import numpy as np
import random

from transformers import (
    AutoModel,
    AutoConfig,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)

from maverick.common.util import *
from maverick.common.constants import *


class MentionClusterClassifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mention_hidden_states, cluster_hidden_states, attention_mask, labels=None):
        # repreated tensor of mention_hs, to append in first position for each possible mention cluster pair
        repeated_mention_hs = mention_hidden_states.unsqueeze(0).repeat(cluster_hidden_states.shape[0], 1, 1)

        # mention cluste pairs by contatenating mention vectors to cluster padded matrix
        mention_cluster_pairs = torch.cat((repeated_mention_hs, cluster_hidden_states), dim=1)
        attention_mask = torch.cat(
            (
                torch.ones(cluster_hidden_states.shape[0], 1, device=self.model.device),
                attention_mask,
            ),
            dim=1,
        )

        logits = self.model(inputs_embeds=mention_cluster_pairs, attention_mask=attention_mask).logits

        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1).to(self.model.device))
        return loss, logits


class Maverick_incr(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # document transformer encoder
        self.encoder_hf_model_name = kwargs["huggingface_model_name"]
        self.encoder = AutoModel.from_pretrained(self.encoder_hf_model_name)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_hf_model_name)
        if kwargs["huggingface_model_name"] == "answerdotai/ModernBERT-base":
            self.encoder.resize_token_embeddings(self.encoder.get_input_embeddings().num_embeddings + 3)
        else:
            self.encoder.resize_token_embeddings(self.encoder.embeddings.word_embeddings.num_embeddings + 3)

        # freeze
        if kwargs["freeze_encoder"]:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # type of representation layer in 'Linear, FC, LSTM-left, LSTM-right, Conv1d'
        self.representation_layer_type = "FC"  # fullyconnected
        # span hidden dimension
        self.token_hidden_size = self.encoder_config.hidden_size
        self.span_pooling = SpanPooling(self.token_hidden_size)
        self.embedding_shape = (20982733, 100)
        entity2id_file = kwargs["entity2id_path"]
        embeddings_file = kwargs["embeddings_path"]
        # Load the mapping dictionary from string IDs to integer indices
        self.entity_to_index = self.load_entity_ids(entity2id_file)
        self._all_entity_ids = list(self.entity_to_index.keys())

        # Load the embeddings as a memmap for efficient, disk-backed access
        self.embeddings = self.load_embeddings(embeddings_file, self.embedding_shape)
        self.kg_embedding_dim = 100
        self.kg_fusion_layer = nn.Linear(self.token_hidden_size * 2 + self.kg_embedding_dim, self.token_hidden_size * 2)
        self.kg_projector = nn.Linear(self.kg_embedding_dim, self.kg_embedding_dim)
        self.entity_linker = SpacyEntityLinkerWrapper()
        # self.kg_projector = nn.Linear(self.kg_embedding_dim, self.token_hidden_size * 2)
        self.gating_fusion = GatingFusion(hidden_dim=self.token_hidden_size * 2, kg_dim=self.kg_embedding_dim)
        

        # if span representation method is to concatenate start and end, a mention hidden size will be 2*token_hidden_size
        if self.encoder_hf_model_name == "answerdotai/ModernBERT-base":
            self.mention_hidden_size = self.token_hidden_size * 3
        else:
            self.mention_hidden_size = self.token_hidden_size * 2   

        # incremental transformer classifier
        self.incremental_model_hidden_size = kwargs.get("incremental_model_hidden_size", 384)  # 768/2
        self.incremental_model_num_layers = kwargs.get("incremental_model_num_layers", 1)
        self.incremental_model_config = DistilBertConfig(num_labels=1, hidden_size=self.incremental_model_hidden_size)
        self.incremental_model = DistilBertForSequenceClassification(self.incremental_model_config).to(self.encoder.device)
        self.incremental_model.distilbert.transformer.layer = self.incremental_model.distilbert.transformer.layer[
            : self.incremental_model_num_layers
        ]
        self.incremental_model.distilbert.embeddings.word_embeddings = None
        self.incremental_transformer = MentionClusterClassifier(model=self.incremental_model)

        # encodes mentions for incremental clustering
        self.incremental_span_encoder = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size * 2,
            output_dim=self.incremental_model_hidden_size,
            hidden_dim=int(self.mention_hidden_size / 2),
        )

        # mention extraction layers
        # representation of start token
        self.start_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        # representation of end token
        self.end_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        # models probability to be the start of a mention
        self.start_token_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
        )

        # model mention probability from start and end representations
        self.start_end_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.mention_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
        )

    def load_entity_ids(self, mapping_file):
        """Loads entity-to-index mapping from a file."""
        entity_to_index = {}
        with open(mapping_file, 'r') as f:
            next(f)  # Skip header if necessary
            for line in f:
                entity, index = line.strip().split('\t')
                entity_to_index[entity] = int(index)
        return entity_to_index

    def load_embeddings(self, embeddings_file, embedding_shape):
        """Loads embeddings as a memory-mapped NumPy array."""
        return np.memmap(embeddings_file, dtype=np.float32, mode='r', shape=embedding_shape)

    def get_embedding(self, entity_str, default_embedding=None):
        """Retrieves the embedding for a given entity string.
        
        Returns a torch tensor on the same device as needed.
        """
        idx = self.entity_to_index.get(entity_str)
        if idx is None:
            # Return a default embedding (e.g., zeros) if the entity isn't found.
            return default_embedding if default_embedding is not None else torch.zeros(self.kg_embedding_dim)
        # Get the embedding row from the memmap and convert it to a torch tensor.
        emb_np = self.embeddings[idx]
        emb_tensor = torch.from_numpy(emb_np)
        return emb_tensor
    
    def augment_mention_reps_with_kg(self, mention_idxs, mention_hidden_states, tokens, bidx=0, combining_method="fusion", use_random_kg_id=False):
        fused_reps = []
        self.epoch_kg_found = 0
        self.epoch_mentions = 0
        mention_hidden_states = mention_hidden_states.squeeze(0)
        kg_enhanced_mention_indices = []
        kg_enhanced_mentions = []
        original_mention_tuples = [tuple(m) for m in mention_idxs.tolist()]

        can_use_random = use_random_kg_id and bool(self._all_entity_ids)
        if use_random_kg_id and not bool(self._all_entity_ids):
             print("Warning: use_random_kg_id is True, but no entity IDs are available for random sampling.")

        # For each predicted mention, convert span indices to a string.
        for i, span in enumerate(original_mention_tuples):
            start_idx, end_idx = span[0], span[1]
            # Here tokens[bidx] is the list of tokens for the document.
            mention_tokens = tokens[bidx][start_idx:end_idx+1]
            mention_text = " ".join(mention_tokens)
            # Entity linking: get entity id (or None if no match)
            entity_id = self.entity_linker.get_entity(mention_text)
            self.epoch_mentions += 1
            
            kg_emb = torch.zeros(self.kg_embedding_dim, device=mention_hidden_states.device) # Default zero
            if entity_id is not None:
                if can_use_random:
                    # Use the random entity ID
                    entity_id = random.choice(self._all_entity_ids)
                retrieved_emb = self.get_embedding(entity_id).to(mention_hidden_states.device)
                # Check if retrieval was successful (might still be zeros if entity not in KG map)
                if not torch.equal(retrieved_emb, torch.zeros(self.kg_embedding_dim, device=mention_hidden_states.device)):
                    kg_emb = retrieved_emb
                    # Track this mention
                    kg_enhanced_mention_indices.append(i)
                    kg_enhanced_mentions.append(span)
                    #print(f"Found KG embedding for span {mention_text} with ID {entity_id}")

            # Count how many KG embeddings were successfully found.
            if not torch.equal(kg_emb, torch.zeros(self.kg_embedding_dim, device=mention_hidden_states.device)):    
                self.epoch_kg_found += 1                

            # Choose the combining strategy.
            if combining_method == "concat":
                # Concatenate the mention representation with the KG embedding.
                combined = torch.cat([mention_hidden_states[i], kg_emb], dim=-1)
                fused_reps.append(combined)
            elif combining_method == "add":
                # Add the KG embedding to the mention representation.
                projected_kg_emb = self.kg_projector(kg_emb)
                combined = mention_hidden_states[i] + projected_kg_emb
                fused_reps.append(combined)
            elif combining_method == "gating":
                current_mention_state = mention_hidden_states[i]
                # GatingFusion expects batch dimension, so unsqueeze inputs
                h_m = current_mention_state.unsqueeze(0) # [1, hidden_dim]
                z_m = kg_emb.unsqueeze(0)                # [1, kg_dim]
                # GatingFusion handles projection internally and returns [1, hidden_dim]
                combined = self.gating_fusion(h_m, z_m).squeeze(0) # Remove batch dim -> [hidden_dim]
                fused_reps.append(combined)    
            elif combining_method == "fusion":
                # Project the fused vector back to the expected dimension.
                #kg_emb = self.kg_projector(kg_emb)
                combined = torch.cat([mention_hidden_states[i], kg_emb], dim=-1)
                fused = self.kg_fusion_layer(combined)
                fused_reps.append(fused)
            elif combining_method == "none":
                # <---- NO KG EMBEDDING AT ALL. Just keep the original mention rep. ---->
                fused_reps.append(mention_hidden_states[i])
            else:
                # Fallback or raise an error if unrecognized method
                raise ValueError(f"Unrecognized combining_method: {combining_method}")



        
        # Print how many were found at the end of an epoch.
        if (self.step + 1) % 79 == 0:
            print(f"Epoch complete: {self.epoch_kg_found} KG embeddings found out of {self.epoch_mentions} mentions")
            self.epoch_kg_found = 0  
            self.epoch_mentions = 0  
        
        if len(fused_reps) > 0:
            fused_reps = torch.stack(fused_reps, dim=0)
        else:
            # In case there are no mentions, return an empty tensor with proper shape.
            fused_reps = torch.empty(0, self.token_hidden_size * 2, device=mention_hidden_states.device)
        
        return fused_reps, kg_enhanced_mentions


    # takes last_hidden_states, eos_mask, ground truth and stage
    def squad_mention_extraction(self, lhs, eos_mask, gold_mentions, gold_starts, stage):
        start_idxs = []
        mention_idxs = []
        start_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        mention_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)

        for bidx in range(0, lhs.shape[0]):
            lhs_batch = lhs[bidx]  # SEQ_LEN X HIDD_DIM
            eos_mask_batch = eos_mask[bidx]  # SEQ_LEN X SEQ_LEN

            # compute start logits
            start_logits_batch = self.start_token_classifier(lhs_batch).squeeze(-1)  # SEQ_LEN

            if gold_starts != None:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(start_logits_batch, gold_starts[bidx])

                # accumulate loss
                start_loss = start_loss + loss

            # compute start positions
            start_idxs_batch = ((torch.sigmoid(start_logits_batch) > 0.5)).nonzero(as_tuple=False).squeeze(-1)

            start_idxs.append(start_idxs_batch.detach().clone())
            # in training, use gold starts to learn to extract mentions, inference use predicted ones
            if stage == "train":
                start_idxs_batch = (
                    ((torch.sigmoid(gold_starts[bidx]) > 0.5)).nonzero(as_tuple=False).squeeze(-1)
                )  # NUM_GOLD_STARTS

            # contains all possible start end indices pairs, i.e. for all starts, all possible ends looking at EOS index
            possibles_start_end_idxs = (eos_mask_batch[start_idxs_batch] == 1).nonzero(as_tuple=False)  # STARTS x 2

            # this is to have reference respect to original positions
            possibles_start_end_idxs[:, 0] = start_idxs_batch[possibles_start_end_idxs[:, 0]]

            possible_start_idxs = possibles_start_end_idxs[:, 0]
            possible_end_idxs = possibles_start_end_idxs[:, 1]

            # extract start and end hidden states
            starts_hidden_states = lhs_batch[possible_end_idxs]  # start
            ends_hidden_states = lhs_batch[possible_start_idxs]  # end

            # Compute the representation for the span.
            if self.encoder_hf_model_name == "answerdotai/ModernBERT-base":
                possible_start_idxs = torch.tensor(possible_start_idxs, device=lhs_batch.device)
                possible_end_idxs = torch.tensor(possible_end_idxs, device=lhs_batch.device)
                MAX_SPAN_LENGTH = 150  # Maximum span length to avoid OOM
                if possible_start_idxs.numel() > 0:
                    num_candidates = possible_start_idxs.size(0)
                    span_lengths = possible_end_idxs - possible_start_idxs + 1
                    span_lengths_clipped = torch.clamp(span_lengths, max=MAX_SPAN_LENGTH)
                    max_span_length = span_lengths_clipped.max().item()
                    range_vec = torch.arange(max_span_length, device=lhs_batch.device).unsqueeze(0)
                    indices = possible_start_idxs.unsqueeze(1) + range_vec
                    valid_mask = indices < (possible_start_idxs.unsqueeze(1) + span_lengths.unsqueeze(1))
                    indices = indices.clamp(max=lhs_batch.size(0) - 1)
                    sub_batch_size = 1024
                    pooled_results = []
                    for i in range(0, num_candidates, sub_batch_size):
                        batch_indices = indices[i : i + sub_batch_size]
                        batch_mask = valid_mask[i : i + sub_batch_size]
                        span_tokens_batch = lhs_batch[batch_indices]
                        pooled = self.span_pooling(span_tokens_batch, mask=batch_mask)
                        pooled_results.append(pooled)
                    span_reps = torch.cat(pooled_results, dim=0)
                else:
                    span_reps = torch.zeros_like(starts_hidden_states)
                s2e_representations = torch.cat((starts_hidden_states, span_reps, ends_hidden_states), dim=-1)
            else:
                s2e_representations = torch.cat(
                    (
                        self.start_token_representation(starts_hidden_states),
                        self.end_token_representation(ends_hidden_states),
                    ),
                    dim=-1,
                )

            # classification of mentions
            s2e_logits = self.start_end_classifier(s2e_representations).squeeze(-1)

            # mention_start_idxs and mention_end_idxs
            mention_idxs.append(possibles_start_end_idxs[torch.sigmoid(s2e_logits) > 0.5].detach().clone())

            if s2e_logits.shape[0] != 0:
                if gold_mentions != None:
                    mention_loss_batch = torch.nn.functional.binary_cross_entropy_with_logits(
                        s2e_logits,
                        gold_mentions[bidx][possible_start_idxs, possible_end_idxs],
                    )
                    mention_loss = mention_loss + mention_loss_batch

        return (start_idxs, mention_idxs, start_loss, mention_loss)

    def incremental_span_clustering(self, mentions_hidden_states, mentions_idxs, gold_clusters, stage):
        pred_cluster_idxs = []  # cluster_idxs = list of list of tuple of offsets (also output) up to mention_idx
        if gold_clusters != None:
            gold_cluster_idxs = unpad_gold_clusters(gold_clusters)  # gold_cluster_idxs, but padded

        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.incremental_model.device)
        mentions_hidden_states = mentions_hidden_states[0]
        idx_to_hs = dict(zip([tuple(m) for m in mentions_idxs.tolist()], mentions_hidden_states))

        # for each mention
        for idx, (
            mention_hidden_states,
            (mention_start_idx, mention_end_idx),
        ) in enumerate(zip(mentions_hidden_states, mentions_idxs)):
            if idx == 0:
                # if first create singleton cluster
                pred_cluster_idxs.append([(mention_start_idx.item(), mention_end_idx.item())])
            else:
                if stage == "train":
                    # if we are in training, retrieve use gold cluster idx to induce loss.
                    cluster_idx, labels = self.new_cluster_idxs_labels(
                        (mention_start_idx, mention_end_idx), gold_cluster_idxs
                    )  # can be used using only tensors
                else:
                    cluster_idx, labels = pred_cluster_idxs, None

                # get cluster padded matrix matrix and attention mask (excludes padding)
                cluster_hs, cluster_am = self.get_cluster_states_matrix(idx_to_hs, cluster_idx, stage)

                # produce logits for each possible cluster mention pair
                mention_cluster_loss, logits = self.incremental_transformer(
                    mention_hidden_states=mention_hidden_states,
                    cluster_hidden_states=cluster_hs,
                    attention_mask=cluster_am,
                    labels=labels,
                )

                if mention_cluster_loss != None:
                    coreference_loss = coreference_loss + mention_cluster_loss

                if stage != "train":
                    # only in inference
                    num_possible_clustering = torch.sum(torch.sigmoid(logits) > 0.5, dim=0).bool().float()

                    if num_possible_clustering == 0:
                        # if no clustering, create new singleton cluster
                        pred_cluster_idxs.append([(mention_start_idx.item(), mention_end_idx.item())])
                    else:
                        # otherwise, take most probabile clustering predicted by the model and assign this mention to that cluster
                        assigned_idx = logits.argmax(axis=0).detach().cpu()
                        pred_cluster_idxs[assigned_idx.item()].append((mention_start_idx.item(), mention_end_idx.item()))
        # normalize loss debug
        if gold_clusters != None:
            coreference_loss = coreference_loss / (mentions_hidden_states.shape[0] if mentions_hidden_states.shape[0] != 0 else 1)

            # coreference_loss = coreference_loss / (len(gold_cluster_idxs) if len(gold_cluster_idxs) != 0 else 1)
        coreferences_pred = [tuple(item) for item in pred_cluster_idxs]  # if len(item) > 1]
        return coreference_loss, coreferences_pred

    def get_cluster_states_matrix(self, idx_to_hs, cluster_idxs, stage):
        # create padded matrix of encoded mentions
        max_length = max([len(x) for x in cluster_idxs])
        if stage == "train":
            max_length = max_length if max_length < 31 else 30
        forward_matrix = torch.zeros(
            (len(cluster_idxs), max_length, self.incremental_model_hidden_size),
            device=self.encoder.device,
        )
        forward_am = torch.zeros((len(cluster_idxs), max_length), device=self.encoder.device)

        for cluster_idx, span_idxs in enumerate(cluster_idxs):
            if stage == "train":
                if len(span_idxs) > 30:
                    span_idxs = sorted(span_idxs)
                    new_idxs = [span_idxs[0]]
                    new_idxs.extend(random.sample(span_idxs, 28))
                    new_idxs.append(span_idxs[-1])
                    span_idxs = new_idxs

            hs = torch.stack([idx_to_hs[span_idx] for span_idx in span_idxs])

            forward_matrix[cluster_idx][: hs.shape[0]] = hs
            forward_am[cluster_idx][: hs.shape[0]] = torch.ones((hs.shape[0]), device=self.encoder.device)

        return forward_matrix, forward_am

    # takes the index of the mention (mention_start, mention_end) and gold coreferences, returns filtered indices (up to mention idx) and labels
    def new_cluster_idxs_labels(self, mention_idxs, gold_coreference_idxs):
        res_coreference_idxs = []
        # list of length number of clusters in gold, and 1.0 where the mention is laying
        labels = [
            1.0 if (mention_idxs[0].item(), mention_idxs[1].item()) in span_idx else 0.0 for span_idx in gold_coreference_idxs
        ]
        # filter cluster up to the mention you are evaluating
        for cluster_idxs in gold_coreference_idxs:
            idxs = []
            for span_idx in cluster_idxs:
                # if span is antecedent to current mention, stay in possible clusters
                if span_idx[0] < mention_idxs[0].item() or (
                    span_idx[0] == mention_idxs[0].item() and span_idx[1] < mention_idxs[1].item()
                ):
                    idxs.append((span_idx[0], span_idx[1]))
            # idxs = sorted(idxs, reverse=True)
            res_coreference_idxs.append(idxs)

        labels = torch.tensor(
            [lab for lab, idx in zip(labels, res_coreference_idxs) if len(idx) != 0],
            device=self.encoder.device,
        )
        res_coreference_idxs = [idx for idx in res_coreference_idxs if len(idx) != 0]
        return res_coreference_idxs, labels

    def get_gold_singleton_labels(self, mention_idxs, raw_gold_clusters):
        gold_singleton_set = set()
        for cluster in raw_gold_clusters:
            if len(cluster) == 1:
                gold_singleton_set.add(tuple(cluster[0]))

        gold_singleton_labels = []
        for (start_i, end_i) in mention_idxs.cpu().tolist():
            if (start_i, end_i) in gold_singleton_set:
                gold_singleton_labels.append(1.0)
            else:
                gold_singleton_labels.append(0.0)

        gold_singleton_labels = torch.tensor(gold_singleton_labels, device=self.encoder.device)
        return gold_singleton_labels

    def build_mention_to_cluster_map(self, raw_gold_clusters):
        mention2cluster = {}
        for cluster in raw_gold_clusters:
            for mention in cluster:
                mention2cluster[tuple(mention)] = cluster
        return mention2cluster

    def get_gold_singletons_for_pred_mentions(self, predicted_mentions, raw_gold_clusters):
        mention2cluster = self.build_mention_to_cluster_map(raw_gold_clusters)

        labels = []
        for (start, end) in predicted_mentions:
            me = (start, end)
            if me in mention2cluster:
                cluster = mention2cluster[me]
                # 1 if that cluster is exactly one mention long, else 0
                labels.append(1 if len(cluster) == 1 else 0)
            else:
                # Not in gold at all => can't be a gold singleton
                labels.append(0)
        return labels
    
    def get_singletons_out_of_coreferences(self, coreferences, predicted_mentions):
        # 1) Build a map from each mention -> cluster for predicted clusters
        #    so we can quickly see how big the predicted cluster is.
        pred_mention_to_cluster = {}
        for cluster in coreferences:  # each 'cluster' is e.g. a list of mention tuples
            for m in cluster:
                pred_mention_to_cluster[m] = cluster

        # 2) Create predicted_singleton_mask (1 or 0 for each mention in mention_idxs)
        pred_singleton_mask = []
        for (start, end) in predicted_mentions:
            mention_tuple = (start, end)
            # If mention not in pred_mention_to_cluster, label=0 (un-clustered mention).
            # If mention is in a predicted cluster of length=1 => 1 else 0
            if mention_tuple in pred_mention_to_cluster:
                if len(pred_mention_to_cluster[mention_tuple]) == 1:
                    pred_singleton_mask.append(1)
                else:
                    pred_singleton_mask.append(0)
            else:
                pred_singleton_mask.append(0)
        return pred_singleton_mask

    def build_mention_representations(self, lhs, mention_idxs):
        # Get start and end indices.
        mention_start_idxs = mention_idxs[:, 0]
        mention_end_idxs = mention_idxs[:, 1]
        
        # Extract corresponding hidden states.
        mentions_start_hidden_states = torch.index_select(lhs, 1, mention_start_idxs)
        mentions_end_hidden_states = torch.index_select(lhs, 1, mention_end_idxs)
        
        if self.encoder_hf_model_name == "answerdotai/ModernBERT-base":
            # Build a pooled span representation between start and end for each mention.
            span_reps_list = []
            for i in range(mention_idxs.shape[0]):
                start_idx = mention_idxs[i, 0].item()
                end_idx = mention_idxs[i, 1].item()
                # Extract the tokens corresponding to the span.
                span_tokens = lhs[0, start_idx:end_idx+1, :]  # shape: (span_length, hidden_size)
                if span_tokens.shape[0] == 0:
                    # If span is empty, use zeros.
                    span_rep = torch.zeros(lhs.size(-1), device=lhs.device)
                else:
                    # Pool over the span tokens.
                    span_tokens = span_tokens.unsqueeze(0)  # add batch dim
                    mask = torch.ones(1, span_tokens.shape[1], device=lhs.device, dtype=torch.bool)
                    span_rep = self.span_pooling(span_tokens, mask=mask).squeeze(0)
                span_reps_list.append(span_rep)
            span_reps = torch.stack(span_reps_list, dim=0)  # (num_mentions, hidden_size)
            # Concatenate start, span, and end.
            mention_reps = torch.cat(
                (mentions_start_hidden_states, span_reps.unsqueeze(0), mentions_end_hidden_states),
                dim=2,
            )
        else:
            # For non-ModernBERT, only use start and end.
            mention_reps = torch.cat((mentions_start_hidden_states, mentions_end_hidden_states), dim=2)
        
        return mention_reps

    def forward(
        self,
        stage,
        input_ids,
        attention_mask,
        eos_mask,
        gold_starts=None,
        gold_mentions=None,
        gold_clusters=None,
        raw_gold_clusters=None,
        tokens=None,
        subtoken_map=None,
        new_token_map=None,
        singletons=True,
        step=None,
    ):
        last_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # B x S x TH
        self.step = step
        lhs = last_hidden_states

        loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        loss_dict = {}
        preds = {}

        (
            start_idxs,
            mention_idxs,
            start_loss,
            mention_loss,
        ) = self.squad_mention_extraction(
            lhs=last_hidden_states,
            eos_mask=eos_mask,
            gold_mentions=gold_mentions,
            gold_starts=gold_starts,
            stage=stage,
        )

        loss_dict["start_loss"] = start_loss
        preds["start_idxs"] = [start.detach().cpu() for start in start_idxs]

        loss_dict["mention_loss"] = mention_loss
        preds["mention_idxs"] = [mention.detach().cpu() for mention in mention_idxs]

        loss = loss + start_loss + mention_loss

        if stage == "train":
            mention_idxs = (gold_mentions[0] == 1).nonzero(as_tuple=False)
        else:
            mention_idxs = mention_idxs[0]

        mention_start_idxs = mention_idxs[:, 0]
        mention_end_idxs = mention_idxs[:, 1]

        mentions_start_hidden_states = torch.index_select(lhs, 1, mention_start_idxs)
        mentions_end_hidden_states = torch.index_select(lhs, 1, mention_end_idxs)

        mentions_hidden_states = torch.cat((mentions_start_hidden_states, mentions_end_hidden_states), dim=2)

        if tokens is not None:
            mentions_hidden_states_kg, kg_enhanced_mentions = self.augment_mention_reps_with_kg(mention_idxs, mentions_hidden_states, tokens, bidx=0)
        mentions_hidden_states_kg = mentions_hidden_states_kg.unsqueeze(0)
        # Build mention representations using the separate function.
        #mentions_hidden_states = self.build_mention_representations(lhs, mention_idxs)

        mentions_hidden_states = self.incremental_span_encoder(mentions_hidden_states_kg)

        coreference_loss, coreferences = self.incremental_span_clustering(
            mentions_hidden_states, mention_idxs, gold_clusters, stage
        )

        loss = loss + coreference_loss
        loss_dict["coreference_loss"] = coreference_loss

        if stage != "train":
            preds["clusters"] = coreferences
            predicted_mentions = mention_idxs.tolist()
            gold_singletons_for_pred = self.get_gold_singletons_for_pred_mentions(predicted_mentions, raw_gold_clusters)
            singleton_preds = self.get_singletons_out_of_coreferences(coreferences, predicted_mentions)   
            preds["singletons"] = singleton_preds
            preds["gold_singletons"] = gold_singletons_for_pred
            preds["kg_enhanced_mentions"] = kg_enhanced_mentions

        loss_dict["full_loss"] = loss
        output = {"pred_dict": preds, "loss_dict": loss_dict, "loss": loss}

        return output
