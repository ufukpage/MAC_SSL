""" Convert txt annotations to json annotations. """
import json


def main():
    ann_path = './dataset/UCF-101/split/trainlist01.txt'
    out_path = './dataset/ucf101/train01.json'
    with open(ann_path, 'r') as f:
        lines = f.read().splitlines()
    anns = []
    for line in lines:
        if line.strip() == '':
            continue
        name, label = line.split(' ')
        anns.append(dict(name=name, label=int(label)))
    with open(out_path, 'w') as f:
        json.dump(anns, f, indent=2)


if __name__ == '__main__':
    main()
