import os
import json
import codecs
from os.path import join
from params import set_params

args = set_params()

def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dump_json(obj, wfname, indent=None):
    with codecs.open(wfname, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


def save_results(names, pubs, results):
    output = {}
    for name in names:
        output[name] = []
        name_pubs = []
        if args.mode == 'train':
            for aid in pubs[name]:
                name_pubs.extend(pubs[name][aid])
        else:
            for pid in pubs[name]:
                name_pubs.append(pid)

        for i in set(results[name]):
            oneauthor = []
            for idx, j in enumerate(results[name]):
                if i == j:
                    oneauthor.append(name_pubs[idx])
            output[name].append(oneauthor)
    
    result_dir = 'out'
    check_mkdir(result_dir)
    result_path = join(result_dir, f'res.json')   

    dump_json(output, result_path, indent=4)
    return result_path