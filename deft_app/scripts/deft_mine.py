import os
import json
import argparse

from deft.discover import DeftMiner

from deft_app.locations import DATA_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use deft to get longforms'
                                     ' associated with shortform')
    parser.add_argument('shortform')
    args = parser.parse_args()
    shortform = args.shortform
    texts_path = os.path.join(DATA_PATH, 'texts', shortform,
                              f'{shortform}_texts.json')
    with open(texts_path, 'r') as f:
        texts = json.load(f)
    texts = texts.values()
    texts = [text for text in texts if text]
    dm = DeftMiner(shortform)
    dm.process_texts(texts)
    longforms = dm.get_longforms()
    out_path = os.path.join(DATA_PATH, 'longforms',
                            f'{shortform}_longforms.json')
    with open(out_path, 'w') as f:
        json.dump(longforms, f)
