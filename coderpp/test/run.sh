mkdir data
mkdir output
python generate_use_data.py --umls_dir ../umls --use_data_dir data
python generate_faiss_index.py --tokenizer_name GanjinZero/coder_eng_pp --model_name_or_path GanjinZero/coder_eng_pp --save_dir data --phrase2idx_path data/phrase2idx.pkl
python confusion_matrix.py --umls_dir ../umls --output_dir output/ --use_data_dir data/ --title coderpp_test

