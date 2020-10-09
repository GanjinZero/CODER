import os


def load_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip().startswith("<e")]

    output_label = []
    output_text = []
    label_set = set()

    for line in lines:
        term_start = line.find(">")
        cui_start = line.find("cui=")
        term = line[term_start + 1:-4]
        cui = line[cui_start + 5:cui_start+13]
        output_label.append(cui)
        output_text.append(term)
        label_set.update([cui])

    return output_text, output_label, label_set

def load(dataset, lang):
    file_path = os.path.join("dataset", dataset + "_GSC_" + lang + "_man.xml")
    output_text, output_label, label_set = load_file(file_path)
    print(dataset, lang)
    print(f"Load count: {len(output_text)}")
    print(f"Different cui: {len(label_set)}")
    return output_text, output_label, label_set

if __name__ == "__main__":
    output_text, output_label, label_set = load("EMEA", "de")
    print(output_text[0:5])
    print(output_label[0:5])
