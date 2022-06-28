mkdir -p ../use_data
mkdir -p ../result

python generate_use_data.py --ner_path ../ner_data/data.txt
python generate_faiss_index.py --model_name_or_path GanjinZero/coder_eng_pp
python clustering.py
python ratio_cut.py
