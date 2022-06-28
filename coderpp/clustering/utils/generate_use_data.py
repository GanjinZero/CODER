# Simply read the NER file and generate idx2phrase, phrase2idx
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ner_path",
        default="../ner_data/Terms_2021_10_07.txt",
        type=str,
        help="Path to NER file"
    )
    args = parser.parse_args()

    phrase_list = []
    print('start loading phrases...')
    with open(args.ner_path, 'r') as f:
        line = f.readline()
        while line:
            line = line.replace("\n", "")
            phrase_list.append(line)
            line = f.readline()
    print('done loading phrases')

    idx2phrase = {idx:phrase for idx, phrase in enumerate(phrase_list)}
    print(type(idx2phrase))
    phrase2idx = {phrase:idx for idx, phrase in enumerate(phrase_list)}
    with open('../use_data/idx2phrase.pkl', 'wb') as f:
        pickle.dump(idx2phrase, f)
    with open('../use_data/phrase2idx.pkl', 'wb') as f:
        pickle.dump(phrase2idx, f)
