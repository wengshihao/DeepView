import json

from tqdm import tqdm

with open('../data/RQ3SSD_VOC000.json') as f1:
    pre_list = json.load(f1)
with open('../data/VOCsel_augGT.json') as f:
    true_list = json.load(f)
print(len(pre_list))
id2scores = {x['image_id']: [] for x in true_list}
print(len(id2scores))
for x in tqdm(pre_list):
    srt = sorted(x['full_score'])
    id2scores[x['image_id']] = id2scores[x['image_id']] + [[srt[-1], srt[-2]]]
results = []
for x in id2scores:
    content_dic = {
        "image_id": x,
        "two_score": id2scores[x]
    }
    results.append(content_dic)
print(len(results))
json_str = json.dumps(results, indent=4)
with open('../data/RQ3SSD_VOCimage.json', 'w') as json_file:
    json_file.write(json_str)
