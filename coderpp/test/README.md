# Steps for testing
`bash run.sh` or 
1. Generate idx2phrase and phrase2idx files for all the phrases we are using for testing. Typically, we are using all cleaned UMLS terms as our train set. `python generate_use_data.py --umls_dir ../umls --use_data_dir data`
2. Generate faiss index for testing. `python generate_faiss_index.py --tokenizer_name GanjinZero/coder_eng_pp --model_name_or_path GanjinZero/coder_eng_pp --save_dir data --phrase2idx_path data/phrase2idx.pkl`
3. Testing. `python confusion_matrix.py --umls_dir ../umls --output_dir output/ --use_data_dir data/ --title coderpp_test`