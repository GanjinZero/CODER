from ndfrt_analysis import get_drug_diseases_to_check
import sys
sys.path.append("../../pretrain")
from load_umls import UMLS

query_to_targets = get_drug_diseases_to_check("may_treat_cui.txt")
query_to_targets_1 = get_drug_diseases_to_check("may_prevent_cui.txt")
cui_set = set()
for query, targets in query_to_targets.items():
    cui_set.update([query])
    cui_set.update(targets)
print(len(cui_set))
umls = UMLS("../../umls", source_range='SNOMEDCT_US')

sty_set = set()
count = 0
for cui in cui_set:
    if cui in umls.cui2sty:
        count += 1
        sty_set.update([umls.cui2sty[cui]])
print(count)
print(len(sty_set))
print(sty_set)

count = 0
for cui in umls.cui2sty:
    if umls.cui2sty[cui] in sty_set:
        count += 1
print(count)
