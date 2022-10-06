
""" Convert txt annotations to json annotations. """
import json
from collections import OrderedDict


def main():
    class_path = 'D:\\UCF-101\\split\\classInd.txt'
    ann_path = 'D:\\UCF-101\\split\\testlist03.txt'
    out_path = 'D:\\UCF-101\\split\\test_split_3.json'
    with open(ann_path, 'r') as f:
        lines = f.read().splitlines()
    class_lu = OrderedDict()
    with open(class_path, 'r') as f:
        class_lines = f.read().splitlines()
        for cl in class_lines:
            ind, name = cl.split(" ")
            class_lu[name] = ind
    anns = []
    for line in lines:
        if line.strip() == '':
            continue
        name, path = line.split('/')
        anns.append(dict(name=line, label=int(class_lu[name])))
    with open(out_path, 'w') as f:
        json.dump(anns, f, indent=2)


if __name__ == '__main__':
    main()
