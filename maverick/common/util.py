import pandas as pd
import spacy
import hydra
import torch
import pytorch_lightning as pl
import numpy as np

from nltk import sent_tokenize
from torch.nn import Module, Linear, LayerNorm, Dropout, GELU, GLU
from torch import nn
import torch.nn.functional as F

from maverick.common.constants import *

class SpacyEntityLinkerWrapper:
    def __init__(self, spacy_model="en_core_web_trf):
        # Load the spaCy language model.
        self.nlp = spacy.load(spacy_model)
        # Add the entity linker pipeline component if not already present.
        if "entityLinker" not in self.nlp.pipe_names:
            self.nlp.add_pipe("entityLinker", last=True)
    
    def get_entity(self, mention_text):
        """
        Process the mention text using spaCy and return the Wikidata ID 
        of the first linked entity in the format expected by the embeddings API.
        """
        doc = self.nlp(mention_text)
        # Check if any linked entities were found.
        if len(doc._.linkedEntities) > 0:
            # For this example, we select the first entity.
            entity = doc._.linkedEntities[0]
            url = entity.get_url()
            # Convert URL from the '/wiki/' format to the '/entity/' format.
            if url.startswith("https://www.wikidata.org/wiki/"):
                url = url.replace("https://www.wikidata.org/wiki/", "http://www.wikidata.org/entity/")
            # Extract the ID which starts with 'Q'
            entity_id = url.split('/')[-1]
            return entity_id
        return None

def get_category_id(mention, antecedent):
    mention, mention_pronoun_id = mention
    antecedent, antecedent_pronoun_id = antecedent

    if mention_pronoun_id > -1 and antecedent_pronoun_id > -1:
        if mention_pronoun_id == antecedent_pronoun_id:
            return CATEGORIES["pron-pron-comp"]
        else:
            return CATEGORIES["pron-pron-no-comp"]

    if mention_pronoun_id > -1 or antecedent_pronoun_id > -1:
        return CATEGORIES["pron-ent"]

    if mention == antecedent:
        return CATEGORIES["match"]

    union = mention.union(antecedent)
    if len(union) == max(len(mention), len(antecedent)):
        return CATEGORIES["contain"]

    return CATEGORIES["other"]


def get_pronoun_id(span):
    if len(span) == 1:
        span = list(span)
        if span[0] in PRONOUNS_GROUPS:
            return PRONOUNS_GROUPS[span[0]]
    return -1


def flatten(l):
    return [item for sublist in l for item in sublist]


def ontonotes_to_dataframe(file_path):
    # read file
    df = pd.read_json(file_path, lines=True)
    # ontonotes is split into words and sentences, jin sentences
    if "sentences" in df.columns:
        df["tokens"] = df["sentences"].apply(lambda x: flatten(x))
    elif "text" in df.columns:  # te
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        texts = df["text"].tolist()
        df["sentences"] = [[[tok.text for tok in s] for s in nlp.pipe(s)] for s in [sent_tokenize(text) for text in texts]]

        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        df["tokens"] = [[tok.text for tok in doc] for doc in nlp.pipe(texts)]
    # compute end of sequence indices
    if "preco" in file_path:
        df["EOS_indices"] = df["tokens"].apply(lambda x: [i + 1 for i, token in enumerate(x) if token == "."])
    else:
        df["EOS_lengths"] = df["sentences"].apply(lambda x: [len(value) for value in x])
        df["EOS_indices"] = df["EOS_lengths"].apply(lambda x: [sum(x[0 : (i[0] + 1)]) for i in enumerate(x)])
    # add speakers
    if "speakers" in df.columns and "wkc" not in file_path and "q" not in file_path:
        df["speakers"] = df["speakers"].apply(lambda x: flatten(x))
    else:
        df["speakers"] = df["tokens"].apply(lambda x: ["-"] * len(x))

    if "clusters" in df.columns:
        df = df[["doc_key", "tokens", "speakers", "clusters", "EOS_indices"]]
    else:
        df = df[["doc_key", "tokens", "speakers", "EOS_indices"]]
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def extract_mentions_to_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    return mention_to_gold


def original_token_offsets(clusters, subtoken_map, new_token_map):
    return [
        tuple(
            [
                (
                    new_token_map[subtoken_map[start]],
                    new_token_map[subtoken_map[end]],
                )
                for start, end in cluster
                if subtoken_map[start] is not None
                and subtoken_map[end] is not None  # only happens first evals, model predicts <s> as mentions
                and new_token_map[subtoken_map[start]]
                is not None  # it happens very rarely that in some weidly formatted sentences the model predicts the speaker name as a possible mention
            ]
        )
        for cluster in clusters
    ]


def unpad_gold_clusters(gold_clusters):
    new_gold_clusters = []
    for batch in gold_clusters:
        new_gold_clusters = []
        for cluster in batch:
            new_cluster = []
            for span in cluster:
                if span[0].item() != -1:
                    new_cluster.append((span[0].item(), span[1].item()))
            if len(new_cluster) != 0:
                new_gold_clusters.append(tuple(new_cluster))
    return new_gold_clusters

class KGEmbeddingTable(nn.Module):
    """
    Wraps the fixed mem‑mapped matrix *and* adds a trainable UNK row.
    """
    def __init__(self, mmap_path: str, num_entities: int, dim: int):
        super().__init__()
        # ➊  fixed embeddings loaded as a non‑trainable buffer
        self.register_buffer(
            "_mmap",
            torch.from_numpy(np.memmap(mmap_path,
                                       dtype=np.float32,
                                       mode="r",
                                       shape=(num_entities, dim)))
        )
        # ➋  one extra row that *does* receive gradients
        self.unk = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.unk, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx ∈ [-1 … num_entities‑1]
        ‑1 maps to UNK.
        """
        unk_mask = (idx == -1)
        # fetch from buffer; whatever is −1 will be clamped to zero row but overwritten later
        idx_safe = idx.clone().clamp(min=0)
        out = self._mmap[idx_safe]
        # replace UNKs
        if unk_mask.any():
            out = out.clone()            # break view on mmap
            out[unk_mask] = self.unk
        return out



class GatingFusion(nn.Module):
    """
    Gating Fusion: 
      h_m: the mention (textual) representation (shape: [batch_size, hidden_dim])
      z_m: the KG embedding (shape: [batch_size, hidden_dim] or 
            shape: [batch_size, kg_dim] projected up to 'hidden_dim')
      
    The module learns a gate g = σ(Wg [h_m; z_m]), and produces:
      h_m_tilde = g ⊙ h_m + (1 - g) ⊙ z_m
    """
    def __init__(self, hidden_dim, kg_dim):
        super().__init__()
        # Linear layer to compute the gating value from concatenated h_m and z_m.
        self.gate_linear = nn.Linear(hidden_dim + kg_dim, hidden_dim)
        # Projection layer to map KG embedding (z_m) to the hidden dimension.
        self.z_project = nn.Linear(kg_dim, hidden_dim)

    def forward(self, h_m, z_m):
        # h_m: [batch_size, hidden_dim]
        # z_m: [batch_size, kg_dim]
        cat = torch.cat([h_m, z_m], dim=-1)  # [batch_size, hidden_dim + kg_dim]
        g = torch.sigmoid(self.gate_linear(cat))  # [batch_size, hidden_dim]
        # Project z_m to hidden_dim
        z_m_proj = self.z_project(z_m)  # [batch_size, hidden_dim]
        h_m_tilde = g * h_m + (1 - g) * z_m_proj
        return h_m_tilde

class SpanPooling(nn.Module):
    def __init__(self, hidden_size):
        super(SpanPooling, self).__init__()
        # This linear layer produces a scalar score for each token
        self.attention_fc = nn.Linear(hidden_size, 1)
    
    def forward(self, token_reps, mask=None):
        # token_reps: [batch, span_length, hidden_size]
        scores = self.attention_fc(token_reps)  # [batch, span_length, 1]
        
        if mask is not None:
            # Set scores for padded tokens (mask == False) to a very negative value,
            # so that their softmax weights become effectively zero.
            scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        attn_weights = torch.nn.functional.softmax(scores, dim=1)   # [batch, span_length, 1]
        pooled = torch.sum(token_reps * attn_weights, dim=1)  # [batch, hidden_size]
        return pooled
    

class FullyConnectedLayer(Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout_prob):
        super(FullyConnectedLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense1 = Linear(self.input_dim, hidden_size)
        self.dense = Linear(hidden_size, self.output_dim)
        self.layer_norm = LayerNorm(hidden_size)
        self.activation_func = GELU()
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense1(temp)
        temp = self.dropout(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dense(temp)
        return temp

    
import torch.utils.checkpoint as checkpoint

class ResidualFullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout_prob):
        super(ResidualFullyConnectedLayer, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_size)
        self.dense = nn.Linear(hidden_size, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Create a projection if dimensions differ.
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward_fn(self, inputs):
        residual = inputs
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dense(x)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        return x + residual

    def forward(self, inputs):
        # Wrap the forward pass with checkpoint to reduce memory usage.
        return checkpoint.checkpoint(self.forward_fn, inputs)

class RepresentationLayer(torch.nn.Module):
    def __init__(self, type, input_dim, output_dim, hidden_dim, **kwargs) -> None:
        super(RepresentationLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.lt = type
        if type == "Linear":
            self.layer = Linear(input_dim, output_dim)
        elif type == "FC":
            self.layer = FullyConnectedLayer(input_dim, output_dim, hidden_dim, dropout_prob=0.2)
        elif type == "LSTM-left":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
        elif type == "LSTM-right":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
        elif type == "LSTM-bidirectional":
            self.layer = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim / 2, bidirectional=True)
        elif type == "Conv1d":
            self.layer = torch.nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=7, stride=1, padding=3)
            self.dropout = Dropout(0.2)
        elif type == "GRU":
            self.layer = torch.nn.GRU(input_size=input_dim, hidden_size=output_dim, bidirectional=False, batch_first=True)
        elif type == "ResidualFC":
            self.layer = ResidualFullyConnectedLayer(input_dim, output_dim, hidden_dim, dropout_prob=0.2)     

    def forward(self, inputs):
        if self.lt == "Linear":
            return self.layer(inputs)
        elif self.lt == "FC":
            return self.layer(inputs)
        elif self.lt == "ResidualFC":
            return self.layer(inputs)
        elif self.lt == "LSTM-left":
            return self.layer(inputs)[0][: self.hidden_dim]
        elif self.lt == "LSTM-right":
            return self.layer(inputs)[0][self.hidden_dim :]
        elif self.lt == "LSTM-bidirectional":
            return self.layer(inputs)[0]
        elif self.lt == "GRU":
            outputs, _ = self.layer(inputs)
            return outputs



def download_load_spacy():
    try:
        import nltk

        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        # colab fix
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
        except:
            nltk.download("punkt")
    except:
        from spacy.cli import download
        import nltk

        nltk.download("punkt")
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
    return nlp
