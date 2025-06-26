<h1 align="center">
  KGE-COREF
</h1>


This repository contains the official implementation of Enhancing Coreference Resolution
with Knowledge Graph Embeddings

---

## 🚀 Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/richtervanemmerik/Msc-AI-Thesis.git
```

### 2. Install Requirements
We recommend using conda for environment management.
```sh
conda env create -f kge_coref.yml
conda activate kge_coref-env
```

### 3. Download SpaCy Language Models

After installing the requirements, you need to download the required SpaCy models:

```sh
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```


### 4. Data 
Corpora official Links:
- [PreCo](https://drive.google.com/file/d/1q0oMt1Ynitsww9GkuhuwNZNq6SjByu-Y/view)
- [LitBank](https://github.com/dbamman/litbank/tree/master/coref/conll)

Since those datasets usually require a preprocessing step to obtain the OntoNotes-like jsonlines format, the original maverick released a ready-to-use version:
https://drive.google.com/drive/u/3/folders/18dtd1Qt4h7vezlm2G0hF72aqFcAEFCUo.

#### Knowledge Graph Embedding Resources

To use knowledge graph embeddings in this project, here are the used resources:

- **OpenKE Toolkit:**  
  [OpenKE Download Page](http://139.129.163.161/index/toolkits)

- **SSL Relation Prediction (Facebook Research):**  
  [GitHub Repository](https://github.com/facebookresearch/ssl-relation-prediction)

These toolkits can be used to generate or donwload the knowledge graph embeddings.




### 5. Hydra
This repository uses [Hydra](https://hydra.cc/) configuration environment.

- In *conf/data/* each yaml file contains a dataset configuration.
- *conf/evaluation/* contains the model checkpoint file path and device settings for model evaluation.
- *conf/logging/* contains details for wandb logging.
- In *conf/model/*, each yaml file contains a model setup.
-  *conf/train/* contains training configurations.
- *conf/root.yaml* regulates the overall configuration of the environment.


### 6. Train
To train a Maverick model, modify *conf/root.yaml* and *conf/model/* with your custom setup. All model configuration can be set in *conf/model/*. Except the SpaCy linker. The default is en_core_web_trf. If you want a different model, change in *src/maverick/common/util.py*.


To train a new model, set the right congfiguration for your setup in *conf/root.yaml* and *conf/model/*, the run the following script:
```
conda activate kge_env
python src/train.py
```
If you encounter a dataset error, the data path files are likely incorrect. Please update the paths to match your system configuration.

### 7. Azure ML

For Azure ML job submission, see [`azure_submit.py`](azure_submit.py) and [`run-maverick-job.sh`](run-maverick-job.sh).

