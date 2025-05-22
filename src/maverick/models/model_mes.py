from curses import raw
import torch
import math
import numpy as np
import httpx
import logging
#logging.getLogger("httpx").setLevel(logging.WARNING)


from torch.nn import init
from torch import nn
from transformers import AutoModel, AutoConfig

from maverick.common.util import *
from maverick.common.constants import *

def weighted_balanced_bce_loss(logits, targets, pos_weight, neg_weight, reduction='mean'):
    # For each sample, use pos_weight if the target is 1, else neg_weight.
    weights = targets * pos_weight + (1 - targets) * neg_weight
    loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction=reduction)
    return loss

def compute_singleton_loss(singleton_scores, gold_singleton_mask):
    label_non_singleton = 1.0 - gold_singleton_mask.float()  # 1 => non-singleton, 0 => singleton
    sig_s = torch.sigmoid(singleton_scores)

    eps = 1e-8
    # per_span_loss = label * -log(1 - sig_s) + (1 - label)* -log(sig_s)
    per_span_loss = label_non_singleton * -torch.log((1 - sig_s).clamp_min(eps)) \
                    + (1 - label_non_singleton) * -torch.log(sig_s.clamp_min(eps))

    return per_span_loss.mean()




class Maverick_mes(torch.nn.Module):
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

        self.singleton_loss = kwargs["singleton_loss"]
        self.singleton_detector = kwargs["singleton_detector"]
        # If using ModernBERT and freezing is enabled, freeze the specified number of layers
        if kwargs.get("freeze_encoder", False) and kwargs["huggingface_model_name"] == "answerdotai/ModernBERT-base":
            self.unfrozen_layers = kwargs["unfrozen_layers"]
            total_layers = len(self.encoder.layers)  # e.g., ModernBERT has 22 layers
            freeze_until = total_layers - self.unfrozen_layers
            print(f"Freezing first {freeze_until} out of {total_layers} layers in ModernBERT")
            for i, layer in enumerate(self.encoder.layers):
                if i < freeze_until:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.mention_treshold = kwargs["mention_treshold"]
        # span representation, now is concat_start_end
        self.span_representation = kwargs["span_representation"]
        # type of representation layer in 'Linear, FC, LSTM-left, LSTM-right, Conv1d'
        self.representation_layer_type = "FC"    
        # span hidden dimension
        self.token_hidden_size = self.encoder_config.hidden_size
        self.span_pooling = SpanPooling(self.token_hidden_size)

        # Define dimension of KG embeddings (32)
        self.kg_embedding_dim = 32
        # Define fusion layer to project concatenated [mention_rep; kg_embedding] back to hidden dim.
        self.kg_fusion_layer = nn.Linear(self.token_hidden_size * 3 + self.kg_embedding_dim, self.token_hidden_size * 3)
        self.entity_linker = SpacyEntityLinkerWrapper()

        # if span representation method is to concatenate start and end, a mention hidden size will be 2*token_hidden_size
        if self.span_representation == "concat_start_end":
            if self.encoder_hf_model_name == "answerdotai/ModernBERT-base":
                self.mention_hidden_size = self.token_hidden_size * 3
            else:
                self.mention_hidden_size = self.token_hidden_size * 2    

        self.num_cats = len(CATEGORIES) + 1  # +1 for ALL
        self.all_cats_size = self.token_hidden_size * self.num_cats
    
        self.coref_start_all_mlps = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.all_cats_size,
            hidden_dim=self.token_hidden_size,
            num_blocks=1,
        )

        self.coref_end_all_mlps = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.all_cats_size,
            hidden_dim=self.token_hidden_size,
            num_blocks=1,
        )

        self.antecedent_s2s_all_weights = nn.Parameter(
            torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
        )
        self.antecedent_e2e_all_weights = nn.Parameter(
            torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
        )
        self.antecedent_s2e_all_weights = nn.Parameter(
            torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
        )
        self.antecedent_e2s_all_weights = nn.Parameter(
            torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
        )

        self.antecedent_s2s_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))
        self.antecedent_e2e_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))
        self.antecedent_s2e_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))
        self.antecedent_e2s_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))

        # mention extraction layers
        # representation of start token
        self.start_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
            num_blocks=1,
        )

        # representation of end token
        self.end_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
            num_blocks=1,
        )

        # models probability to be the start of a mention
        self.start_token_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
            num_blocks=1,
        )

        # model mention probability from start and end representations
        self.start_end_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.mention_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
            num_blocks=1,
        )
        # for every candidate mention, predicts whether it is a singleton
        self.singleton_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.mention_hidden_size,  # expects the mention representation dimension
            output_dim=1,
            hidden_dim=self.token_hidden_size,
            num_blocks=1,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        W = [
            self.antecedent_s2s_all_weights,
            self.antecedent_e2e_all_weights,
            self.antecedent_s2e_all_weights,
            self.antecedent_e2s_all_weights,
        ]

        B = [
            self.antecedent_s2s_all_biases,
            self.antecedent_e2e_all_biases,
            self.antecedent_s2e_all_biases,
            self.antecedent_e2s_all_biases,
        ]

        for w, b in zip(W, B):
            init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(b, -bound, bound)


    def augment_mention_reps_with_kg(self, mention_idxs, mention_reps, tokens, bidx=0):
        fused_reps = []
        # For each predicted mention, convert span indices to a string.
        for i, span in enumerate(mention_idxs.tolist()):
            start_idx, end_idx = span[0], span[1]
            # Here tokens[bidx] is the list of tokens for the document.
            mention_tokens = tokens[bidx][start_idx:end_idx+1]
            mention_text = " ".join(mention_tokens)
            # Entity linking: get entity id (or None if no match)
            entity_id = self.entity_linker.get_entity(mention_text)
            if entity_id is not None:
                kg_emb = self.fetch_kg_embedding(entity_id)
            else:
                kg_emb = torch.zeros(self.kg_embedding_dim, device=mention_reps.device)
            if not torch.equal(kg_emb, torch.zeros(self.kg_embedding_dim, device=mention_reps.device)):    
                print(f"Entity ID for '{mention_text}': {entity_id}")
                print(f"KG embedding for '{mention_text}': {kg_emb}")
            # Fuse the original mention rep with the KG embedding (concatenation)
            # Here mention_reps[i] is the textual representation of the mention.
            combined = torch.cat([mention_reps[i], kg_emb], dim=-1)
            # Project the fused vector back to the expected dimension.
            fused = self.kg_fusion_layer(combined)
            fused_reps.append(fused)
        if len(fused_reps) > 0:
            fused_reps = torch.stack(fused_reps, dim=0)
        else:
            # In case there are no mentions, return an empty tensor with proper shape.
            fused_reps = torch.empty(0, self.token_hidden_size * 2, device=mention_reps.device)
        return fused_reps

    # takes last_hidden_states, eos_mask, ground truth and stage
    def eos_mention_extraction(self, lhs, eos_mask, gold_mentions, gold_starts, stage):
        start_idxs = []
        mention_idxs = []
        start_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        mention_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)

        for bidx in range(lhs.shape[0]):
            lhs_batch = lhs[bidx]  # SEQ_LEN x HIDDEN_DIM
            eos_mask_batch = eos_mask[bidx]  # SEQ_LEN x SEQ_LEN

            # Compute logits and candidate scores for start tokens.
            start_logits_batch = self.start_token_classifier(lhs_batch).squeeze(-1)  # SEQ_LEN
            candidate_scores = torch.sigmoid(start_logits_batch)

            if gold_starts is not None:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(start_logits_batch, gold_starts[bidx])
                start_loss = start_loss + loss


            # Select candidate start indices based on the threshold.
            start_idxs_batch = (candidate_scores > self.mention_treshold).nonzero(as_tuple=False).squeeze(-1)
            
            start_idxs.append(start_idxs_batch.detach().clone())

            
            if stage == "train" and gold_starts is not None:
                start_idxs_batch = (torch.sigmoid(gold_starts[bidx]) > 0.5).nonzero(as_tuple=False).squeeze(-1)

            # Get all possible start-end indices from the candidate start indices.
            possibles_start_end_idxs = (eos_mask_batch[start_idxs_batch] == 1).nonzero(as_tuple=False)
            possibles_start_end_idxs[:, 0] = start_idxs_batch[possibles_start_end_idxs[:, 0]]
            possible_start_idxs = possibles_start_end_idxs[:, 0]
            possible_end_idxs = possibles_start_end_idxs[:, 1]

            # Extract the hidden states for these candidate spans.
            starts_hidden_states = lhs_batch[possible_end_idxs]
            ends_hidden_states = lhs_batch[possible_start_idxs]

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

            s2e_logits = self.start_end_classifier(s2e_representations).squeeze(-1)
            mention_idxs.append(possibles_start_end_idxs[(torch.sigmoid(s2e_logits) > self.mention_treshold)].detach().clone())

            if s2e_logits.shape[0] != 0 and stage != "test" and gold_mentions is not None:
                mention_loss_batch = torch.nn.functional.binary_cross_entropy_with_logits(
                    s2e_logits,
                    gold_mentions[bidx][possible_start_idxs, possible_end_idxs],
                )
                mention_loss = mention_loss + mention_loss_batch

        return (start_idxs, mention_idxs, start_loss, mention_loss, s2e_representations)


    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k), device="cpu")
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(
            zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)
        ):
            gold_clusters = self.extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.encoder.device)
        return new_cluster_labels

    def _get_all_labels(self, clusters_labels, categories_masks):

        categories_labels = clusters_labels.unsqueeze(1).repeat(1, self.num_cats, 1, 1) * categories_masks
        all_labels = categories_labels

        return all_labels

    def mes_span_clustering(
        self, mention_start_reps, mention_end_reps, mention_start_idxs, mention_end_idxs, gold, stage, mask, add, sing
    ):
        if mention_start_reps[0].shape[0] == 0:
            return torch.tensor([0.0], requires_grad=True, device=self.encoder.device), []
        coref_logits = self._calc_coref_logits(mention_start_reps, mention_end_reps)
        coref_logits = coref_logits[0] * mask[0]
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)

        coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits]).unsqueeze(0)
        if stage in ["train", "val"]:
            labels = self._get_cluster_labels_after_pruning(mention_start_idxs, mention_end_idxs, gold)
            all_labels = self._get_all_labels(labels, mask)
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, all_labels)

        coref_logits = coref_logits.sum(dim=1)
        doc, m2a, singletons = self.create_mention_to_antecedent_singletons(mention_start_idxs, mention_end_idxs, coref_logits)
        if not sing:
            singletons = []
        coreferences = self.create_clusters(m2a, singletons, add)
        return coreference_loss, coreferences

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_cats, self.token_hidden_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # bnkf/bnlg

    def _calc_coref_logits(self, top_k_start_coref_reps, top_k_end_coref_reps):
        all_starts = self.transpose_for_scores(self.coref_start_all_mlps(top_k_start_coref_reps))
        all_ends = self.transpose_for_scores(self.coref_end_all_mlps(top_k_end_coref_reps))

        logits = (
            torch.einsum("bnkf, nfg, bnlg -> bnkl", all_starts, self.antecedent_s2s_all_weights, all_starts)
            + torch.einsum("bnkf, nfg, bnlg -> bnkl", all_ends, self.antecedent_e2e_all_weights, all_ends)
            + torch.einsum("bnkf, nfg, bnlg -> bnkl", all_starts, self.antecedent_s2e_all_weights, all_ends)
            + torch.einsum("bnkf, nfg, bnlg -> bnkl", all_ends, self.antecedent_e2s_all_weights, all_starts)
        )

        biases = (
            torch.einsum("bnkf, nf -> bnk", all_starts, self.antecedent_s2s_all_biases).unsqueeze(-2)
            + torch.einsum("bnkf, nf -> bnk", all_ends, self.antecedent_e2e_all_biases).unsqueeze(-2)
            + torch.einsum("bnkf, nf -> bnk", all_ends, self.antecedent_s2e_all_biases).unsqueeze(-2)
            + torch.einsum("bnkf, nf -> bnk", all_starts, self.antecedent_e2s_all_biases).unsqueeze(-2)
        )

        return logits + biases

    def _get_categories_labels(self, tokens, subtoken_map, new_token_map, span_starts, span_ends):
        max_k = span_starts.shape[0]

        doc_spans = []
        for start, end in zip(span_starts, span_ends):
            token_indices = [new_token_map[0][idx] for idx in set(subtoken_map[0][start : end + 1]) if idx is not None and idx >= 0 and idx < len(new_token_map[0])]
            span = {tokens[0][idx].lower() for idx in token_indices if idx is not None}
            pronoun_id = get_pronoun_id(span)
            doc_spans.append((span - STOPWORDS, pronoun_id))

        categories_labels = np.zeros((max_k, max_k)) - 1
        for i in range(max_k):
            for j in list(range(max_k))[:i]:
                categories_labels[i, j] = get_category_id(doc_spans[i], doc_spans[j])

        categories_labels = torch.tensor(categories_labels, device=self.encoder.device).unsqueeze(0)
        categories_masks = [categories_labels == cat_id for cat_id in range(self.num_cats - 1)] + [categories_labels != -1]
        categories_masks = torch.stack(categories_masks, dim=1).int()
        return categories_labels, categories_masks

    def create_mention_to_antecedent_singletons(self, span_starts, span_ends, coref_logits):
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        bs, n_spans, _ = coref_logits.shape
        # long distance regularization
        # a = torch.sigmoid(coref_logits)
        # m1 = torch.arange(coref_logits.shape[1], device=a.device).unsqueeze(0).repeat(coref_logits.shape[1], 1)
        # m2 = m1.transpose(0, 1)
        # m = (torch.ones_like(coref_logits[0], device=a.device) + (m2 - m1 - 1) / 1000).tril().fill_diagonal_(0).unsqueeze(0)
        # no_ant = 1 - torch.sum(a - a * m > 0.5, dim=-1).bool().float()
        # coref_logits = torch.cat((coref_logits - coref_logits * m, no_ant.unsqueeze(-1)), dim=-1)
        no_ant = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # [batch_size, max_k, max_k + 1]
        coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)

        span_starts = span_starts.detach().cpu()
        span_ends = span_ends.detach().cpu()
        max_antecedents = coref_logits.argmax(axis=-1).detach().cpu()
        doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
        # indices where antecedent is not null.
        mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

        antecedent_indices = max_antecedents[max_antecedents < n_spans]
        span_indices = np.stack([span_starts.detach().cpu(), span_ends.detach().cpu()], axis=-1)

        mentions = span_indices[doc_indices, mention_indices]
        antecedents = span_indices[doc_indices, antecedent_indices]
        non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]

        sing_indices = np.zeros_like(len(np.setdiff1d(non_mentions, antecedent_indices)))
        singletons = span_indices[sing_indices, np.setdiff1d(non_mentions, antecedent_indices)]

        # mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1 and len(antecedents.shape) == 1:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=0)
        else:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1:
            mention_to_antecedent = [mention_to_antecedent]

        if len(singletons.shape) == 1:
            singletons = [singletons]

        return doc_indices, mention_to_antecedent, singletons

    def create_clusters(self, m2a, singletons, add):
        # Note: mention_to_antecedent is a numpy array
        if add != None:
            clusters = add
            mention_to_cluster = {m: i for i, c in enumerate(clusters) for m in c}
        else:
            clusters, mention_to_cluster = [], {}
        for mention, antecedent in m2a:
            mention, antecedent = tuple(mention), tuple(antecedent)
            if antecedent in mention_to_cluster:
                cluster_idx = mention_to_cluster[antecedent]
                if mention not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(mention)
                    mention_to_cluster[mention] = cluster_idx
            elif mention in mention_to_cluster:
                cluster_idx = mention_to_cluster[mention]
                if antecedent not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(antecedent)
                    mention_to_cluster[antecedent] = cluster_idx
            else:
                cluster_idx = len(clusters)
                mention_to_cluster[mention] = cluster_idx
                mention_to_cluster[antecedent] = cluster_idx
                clusters.append([antecedent, mention])

        clusters = [tuple(cluster) for cluster in clusters]
        # maybe order stuff?
        if len(singletons) != 0:
            clust = []
            while len(clusters) != 0 or len(singletons) != 0:
                if len(singletons) == 0:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
                elif len(clusters) == 0:
                    clust.append(tuple([tuple(singletons[0])]))
                    singletons = singletons[1:]
                elif singletons[0][0] < sorted(clusters[0], key=lambda x: x[0])[0][0]:
                    clust.append(tuple([tuple(singletons[0])]))
                    singletons = singletons[1:]
                else:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
            return clust
        return clusters

    def extract_clusters(self, gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in cluster if (-1) not in m) for cluster in gold_clusters]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
        return gold_clusters

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

    def forward(
        self,
        stage,
        input_ids,
        attention_mask,
        eos_mask,
        gold_starts=None,
        raw_gold_clusters=None,
        gold_mentions=None,
        gold_clusters=None,
        tokens=None,
        subtoken_map=None,
        new_token_map=None,
        add=None,
        singletons=True,
        longdoc=None,
        **kwargs
    ):

        loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        loss_dict = {}
        preds = {}

        if longdoc == None:
            last_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[
                "last_hidden_state"
            ]  # B x S x TH

            lhs = last_hidden_states

            (
                start_idxs,
                mention_idxs,
                start_loss,
                mention_loss,
                enriched_mention_reps,
            ) = self.eos_mention_extraction(
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
            elif stage == "test" and gold_mentions != None:
                mention_idxs = (gold_mentions[0] == 1).nonzero(as_tuple=False)
            else:
                mention_idxs = mention_idxs[0]
            ### needed for singleton detection, otherwise not needed
            if self.encoder_hf_model_name == "answerdotai/ModernBERT-base" and self.singleton_detector:
                # Clamp indices to ensure they are within the bounds of enriched_mention_reps.
                max_idx = enriched_mention_reps.size(0)
                # Clamp the indices in the first column (which are used for extraction)
                clamped_indices = mention_idxs[:, 0].clamp(max=max_idx - 1)
                selected_mention_reps = enriched_mention_reps[clamped_indices]
                mention_reps = selected_mention_reps
            else:
                mention_start_idxs = mention_idxs[:, 0]
                mention_end_idxs = mention_idxs[:, 1]
                mentions_start_hidden_states = torch.index_select(lhs, 1, mention_start_idxs)
                mentions_end_hidden_states = torch.index_select(lhs, 1, mention_end_idxs)
                mention_reps = torch.cat((mentions_start_hidden_states, mentions_end_hidden_states), dim=-1)

            if tokens is not None:
                augmented_reps = self.augment_mention_reps_with_kg(mention_idxs, enriched_mention_reps, tokens, bidx=0)
            mention_start_idxs = mention_idxs[:, 0]
            mention_end_idxs = mention_idxs[:, 1]
            mentions_start_hidden_states = torch.index_select(lhs, 1, mention_start_idxs)
            mentions_end_hidden_states = torch.index_select(lhs, 1, mention_end_idxs)
        elif longdoc == "naive":
            mention_idxs, mentions_start_hidden_states, mentions_end_hidden_states = self.naive_long_doc_forward(
                stage, input_ids, attention_mask, eos_mask, gold_starts, gold_mentions
            )
            mention_idxs = mention_idxs[0]
        elif longdoc == "proper":
            pass
        else:
            raise Exception("unsupported")


        mention_start_idxs = mention_idxs[:, 0]
        mention_end_idxs = mention_idxs[:, 1]

        _, categories_masks = self._get_categories_labels(
            tokens, subtoken_map, new_token_map, mention_start_idxs, mention_end_idxs
        )
        augmented_start_reps = augmented_reps[:, :self.token_hidden_size] 
        augmented_end_reps   = augmented_reps[:, 2*self.token_hidden_size:3*self.token_hidden_size]
        augmented_start_reps = augmented_start_reps.unsqueeze(0)
        augmented_end_reps = augmented_end_reps.unsqueeze(0)
        coreference_loss, coreferences = self.mes_span_clustering(
            augmented_start_reps,
            augmented_end_reps,
            mention_start_idxs,
            mention_end_idxs,
            gold_clusters,
            stage,
            categories_masks,
            add,
            singletons,
        )

        if stage == "train":
            loss = loss + coreference_loss
        loss_dict["coreference_loss"] = coreference_loss

        if self.singleton_detector:
            singleton_logits = self.singleton_classifier(mention_reps).squeeze(-1)
        if self.singleton_loss:
            if stage == "train" and raw_gold_clusters is not None:
                singleton_labels = self.get_gold_singleton_labels(mention_idxs, raw_gold_clusters)
                #singleton_loss = focal_loss_with_logits(singleton_logits, singleton_labels, gamma=2.0, alpha=0.25)
                singleton_loss = weighted_balanced_bce_loss(singleton_logits, singleton_labels, pos_weight=2.0, neg_weight=1.0)
                #singleton_loss = torch.nn.functional.binary_cross_entropy_with_logits(singleton_logits, singleton_labels)
                #singleton_loss = compute_singleton_loss(singleton_logits, singleton_labels)
                lambda_singleton = 0.2
                loss = loss + lambda_singleton * singleton_loss
                loss_dict["singleton_loss"] = singleton_loss


        if stage != "train":
            preds["clusters"] = coreferences
            predicted_mentions = mention_idxs.tolist()
            gold_singletons_for_pred = self.get_gold_singletons_for_pred_mentions(predicted_mentions, raw_gold_clusters)
            singleton_preds = self.get_singletons_out_of_coreferences(coreferences, predicted_mentions)    
            if self.singleton_detector:
                singleton_preds = (torch.sigmoid(singleton_logits) > 0.5).long()
                            
            preds["singletons"] = singleton_preds
            preds["gold_singletons"] = gold_singletons_for_pred

        loss_dict["full_loss"] = loss
        output = {"pred_dict": preds, "loss_dict": loss_dict, "loss": loss}

        return output

    def naive_long_doc_forward(
        self,
        stage,
        input_ids,
        attention_mask,
        eos_mask,
        gold_starts=None,
        gold_mentions=None,
        max_seq_len=2000,
    ):
        length = input_ids.shape[1]
        slices_seq_index = [
            (eos_mask[0][step] == 1).nonzero(as_tuple=False)[-1][0].item() for step in range(max_seq_len, length, max_seq_len)
        ]
        star = (eos_mask[0][0] == 1).nonzero(as_tuple=False)[-1][0].item()
        if len(slices_seq_index) == 0 or slices_seq_index[-1] != length:
            slices_seq_index.append(length)
        prev = 0
        start_idxs = []
        mention_idxs = []
        index_start_hidden_states = []
        index_end_hidden_states = []
        for index in slices_seq_index:
            if prev != 0:
                index_input_ids = torch.cat((input_ids[0][0:star], input_ids[0][prev:index]), dim=0).unsqueeze(0)
                index_attention_mask = torch.cat((attention_mask[0][0:star], attention_mask[0][prev:index]), dim=0).unsqueeze(0)
                index_eos_mask = torch.cat((eos_mask[0][0:star], eos_mask[0][prev:index]), dim=0).unsqueeze(0)
                index_eos_mask = torch.cat((index_eos_mask[0][:, 0:star], index_eos_mask[0][:, prev:index]), dim=1).unsqueeze(0)
                if gold_mentions != None:
                    index_gold_mentions = torch.cat((gold_mentions[0][0:star], gold_mentions[0][prev:index]), dim=0).unsqueeze(0)
                    index_gold_mentions = torch.cat(
                        (index_gold_mentions[0][:, 0:star], index_gold_mentions[0][:, prev:index]), dim=1
                    ).unsqueeze(0)
                    index_gold_starts = torch.cat((gold_starts[0][0:star], input_ids[0][prev:index]), dim=0).unsqueeze(0)
                else:
                    index_gold_mentions = gold_mentions
                    index_gold_starts = gold_starts
            else:
                index_input_ids = input_ids[0][prev:index].unsqueeze(0)
                index_attention_mask = attention_mask[0][prev:index].unsqueeze(0)
                index_eos_mask = eos_mask[0][prev:index].unsqueeze(0)
                index_eos_mask = index_eos_mask[0][:, prev:index].unsqueeze(0)
                if gold_mentions != None:
                    index_gold_mentions = gold_mentions[0][prev:index].unsqueeze(0)
                    index_gold_mentions = index_gold_mentions[0][:, prev:index].unsqueeze(0)
                    index_gold_starts = gold_starts[0][prev:index].unsqueeze(0)
                else:
                    index_gold_mentions = gold_mentions
                    index_gold_starts = gold_starts

            index_last_hidden_states = self.encoder(input_ids=index_input_ids, attention_mask=index_attention_mask)[
                "last_hidden_state"
            ]  # B x S x TH

            (
                index_start_idxs,
                index_mention_idxs,
                _,
                _,
            ) = self.eos_mention_extraction(
                lhs=index_last_hidden_states,
                eos_mask=index_eos_mask,
                gold_mentions=index_gold_mentions,
                gold_starts=index_gold_starts,
                stage=stage,
            )

            if index_gold_mentions != None:
                index_mention_idxs = [(index_gold_mentions[0] == 1).nonzero(as_tuple=False)]
            if prev != 0:
                start_idxs.append(torch.add(index_start_idxs[0][index_start_idxs[0] > star], prev - star))
                mention_idxs.append(torch.add(index_mention_idxs[0][index_mention_idxs[0][:, 0] > star], prev - star))
                index_start_hidden_states.append(
                    torch.index_select(
                        index_last_hidden_states, 1, index_mention_idxs[0][index_mention_idxs[0][:, 0] > star][:, 0]
                    )
                )
                index_end_hidden_states.append(
                    torch.index_select(
                        index_last_hidden_states, 1, index_mention_idxs[0][index_mention_idxs[0][:, 0] > star][:, 1]
                    )
                )

            else:
                start_idxs.append(index_start_idxs[0])
                idxs = index_mention_idxs[0]
                mention_idxs.append(idxs)
                index_start_hidden_states.append(torch.index_select(index_last_hidden_states, 1, idxs[:, 0]))
                index_end_hidden_states.append(torch.index_select(index_last_hidden_states, 1, idxs[:, 1]))

            prev = index

        mentions_start_hidden_states = torch.cat(index_start_hidden_states, dim=1)
        mentions_end_hidden_states = torch.cat(index_end_hidden_states, dim=1)
        mention_idxs = [torch.cat(mention_idxs, dim=0)]
        return mention_idxs, mentions_start_hidden_states, mentions_end_hidden_states


