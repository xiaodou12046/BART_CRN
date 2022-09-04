import json
import os
import re
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='rest_acos', choices=['rest_acos', 'lap_acos'])
    args = parser.parse_args()

    if args.dataset == 'rest_acos':
        data_paths = [f'./Restaurant-ACOS-origin/{f}' for f in os.listdir('./Restaurant-ACOS-origin') if re.search(r'.\.tsv$', f)]
    else:
        data_paths = [f'./Laptop-ACOS-origin/{f}' for f in os.listdir('./Laptop-ACOS-origin') if re.search(r'.\.tsv$', f)]

    category_set = set()
    sentiment_dic = {'0': 'NEG', '1': 'NEU', '2': 'POS'}
    for path in data_paths:
        data = []
        with open(path, 'r', encoding='utf-8') as f1:
            for line in f1.readlines():
                dic = {}
                l = line.strip().split('\t')
                sentence, quads = l[0], l[1:]
                sentence_list = sentence.split()

                dic["raw_words"] = sentence
                dic["words"] = sentence_list
                dic["aspects"] = []
                dic["opinions"] = []


                for quad in quads:
                    quad_list = re.split(r'[\s|,]', quad)

                    asp_start, asp_end, asp_category = int(quad_list[0]), int(quad_list[1]), quad_list[2]
                    aspect = {
                        "from": asp_start,
                        "to": asp_end,
                        "category": asp_category,
                        "term": [] if asp_start == -1 else sentence_list[asp_start:asp_end]
                    }
                    dic['aspects'].append(aspect)
                    category_set.add(asp_category)

                    op_start, op_end, op_sentiment = int(quad_list[4]), int(quad_list[5]), quad_list[3]
                    opinion = {
                        "from": op_start,
                        "to": op_end,
                        "sentiment": sentiment_dic[op_sentiment],
                        "term": [] if op_start == -1 else sentence_list[op_start:op_end]
                    }
                    dic['opinions'].append(opinion)

                data.append(dic)


        save_path = args.dataset + '/' + re.sub(r'(\./.*/)|(\.tsv)', '', path) + '_convert.json'
        with open(save_path, 'w', encoding='utf-8') as f2:
            f2.write(json.dumps(data, indent=2))

    print(len(category_set))
    with open(f'{args.dataset}/category.json', 'w', encoding='utf-8') as f3:
        f3.write(json.dumps(list(category_set), indent=2))





