:<<!
python umnsrs/umnsrs_eval.py /home/yz/pretraining_models/biobert_v1.1
python umnsrs/umnsrs_eval.py /home/yz/pretraining_models/bert-base-cased
python umnsrs/umnsrs_eval.py /home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
python umnsrs/umnsrs_eval.py /home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract
python umnsrs/umnsrs_eval.py /home/yz/pretraining_models/kexinghuang_clinical
python umnsrs/umnsrs_eval.py emilyalsentzer/Bio_ClinicalBERT
python umnsrs/umnsrs_eval.py ../models/2020_eng
python umnsrs/umnsrs_eval.py ../models/2020_all
!
python umnsrs/umnsrs_eval.py ../embeddings/GoogleNews-vectors-negative300.bin
python umnsrs/umnsrs_eval.py ../embeddings/wikipedia-pubmed-and-PMC-w2v.bin
python umnsrs/umnsrs_eval.py ../embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin
python umnsrs/umnsrs_eval.py ../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin
:<<!
python mayosrs/srs_eval.py /home/yz/pretraining_models/biobert_v1.1 bert
python mayosrs/srs_eval.py /home/yz/pretraining_models/bert-base-cased bert
python mayosrs/srs_eval.py /home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext bert
python mayosrs/srs_eval.py /home/yz/pretraining_models/BiomedNLP-PubMedBERT-base-uncased-abstract 
python mayosrs/srs_eval.py /home/yz/pretraining_models/kexinghuang_clinical bert
python mayosrs/srs_eval.py emilyalsentzer/Bio_ClinicalBERT bert
python mayosrs/srs_eval.py ../models/2020_eng bert
python mayosrs/srs_eval.py ../models/2020_all bert
!
:<<!
python mayosrs/srs_eval.py /home/yz/pretraining_models/cui2vec.pkl cui
python mayosrs/srs_eval.py ../embeddings/GoogleNews-vectors-negative300.bin word
python mayosrs/srs_eval.py ../embeddings/wikipedia-pubmed-and-PMC-w2v.bin word
python mayosrs/srs_eval.py ../embeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin word
python mayosrs/srs_eval.py ../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin word
!
#python mayosrs/srs_eval.py ../embeddings/DeVine_etal_200.txt cui
#python mayosrs/srs_eval.py ../embeddings/claims_codes_hs_300.txt cui