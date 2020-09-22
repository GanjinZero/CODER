from sklearn.model_selection import train_test_split


def load_all_relation():
    with open("./data/relation_all.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    cui1_list = []
    rel_list = []
    cui2_list = []
    for line in lines:
        cui1, rel, cui2, source = line.strip().split("|")
        if source == "diseasedb":
            cui1_list.append(cui1)
            cui2_list.append(cui2)
            rel_list.append(rel)

    print("Tri group count:", len(cui1_list))
    print("Relation count:", len(set(rel_list)))
    print(set(rel_list))

    return cui1_list, cui2_list, rel_list


def split_and_save():
    cui1_list, cui2_list, rel_list = load_all_relation()
    x = [[cui1_list[i], cui2_list[i]] for i in range(len(cui1_list))]
    x_train, x_test, y_train, y_test = train_test_split(
        x, rel_list, test_size=0.2, random_state=72, stratify=rel_list)
    with open("./data/x_train.txt", "w", encoding="utf-8") as f:
        for x in x_train:
            f.write(x[0] + "\t" + x[1] + "\n")
    with open("./data/x_test.txt", "w", encoding="utf-8") as f:
        for x in x_test:
            f.write(x[0] + "\t" + x[1] + "\n")
    with open("./data/y_train.txt", "w", encoding="utf-8") as f:
        for y in y_train:
            f.write(y + "\n")
    with open("./data/y_test.txt", "w", encoding="utf-8") as f:
        for y in y_test:
            f.write(y + "\n")

if __name__ == "__main__":
    #split_and_save()
    load_all_relation()
