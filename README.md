# CODER
CODER: Knowledge infused cross-lingual medical term embedding for term normalization. [Paper](http://arxiv.org/abs/2011.02947)

# Use the model by transformers
Models have been uploaded to huggingface/transformers repo.

## To use CODER_ENG
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
```

## To use CODER_ALL (Multi-Language)
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ALL")
model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ALL")
```

# Train your model
```shell
cd pretrain
python train.py --umls_dir your_umls_dir --model_name_or_path monologg/biobert_v1.1_pubmed
```
To notice, if you are using Windows system.
```shell
cd pretrain
python train.py --umls_dir your_umls_dir --model_name_or_path monologg/biobert_v1.1_pubmed --num_workers 0
```
your_umls_dir should contain **MRCONSO.RRF**, **MRREL.RRF** and **MRSTY.RRF**.
UMLS Download path:[UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2020AA).

# A small tool for load UMLS RRF

```python
from pretrain.load_umls import UMLS
umls = UMLS(your_umls_dir)
```

# Test CODER or other embeddings
## CADEC
```shell
cd test
python cadec/cadec_eval.py bert_model_name_or_path
python cadec/cadec_eval.py word_embedding_path
```

## MayoSRS & UMNSRS
```shell
cd test
python umnsrs/umnsrs_eval.py bert_model_name_or_path
```

```shell
cd test
python mayosrs/srs_eval.py bert_model_name_or_path bert
python mayosrs/srs_eval.py word_embedding_path word
python mayosrs/srs_eval.py cui_embedding_path cui
```
For testing mayosrs, you should specify the embedding type.

## MCSM & MRM
```shell
cd test/embeddings_reimplement
python mcsm.py
python codes_analysis.py # MRM_CCS
python ndfrt_analysis.py # MRM_NDF_RT
```

## Entity pairs relation classification
Only sample datas are provided.
```shell
cd test/diseasedb
python train.py your_embedding embedding_type freeze_or_not gpu_id
```
- your_embedding should contain an embedding or a pre-trained model
- embedding_type should be in [bert, word, cui]
- freeze_or_not should be in [T, F], T means freeze the embedding, and F means fine-tune the embedding
- gpu_id should be a number, e.g. 0

## mantra
Download [the Mantra GSC](https://files.ifi.uzh.ch/cl/mantra/gsc/GSC-v1.1.zip) and unzip the xml files to /test/mantra/dataset, run
```
cd test/mantra
python test.py
```

## Citation
```bibtex
@article{yuan2020coder,
  title={CODER: Knowledge infused cross-lingual medical term embedding for term normalization},
  author={Yuan, Zheng and Zhao, Zhengyun and Yu, Sheng},
  journal={arXiv preprint arXiv:2011.02947},
  year={2020}
}
```
