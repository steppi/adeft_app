import os
import json
import argparse

from adeft.discover import DeftMiner

from adeft_app.locations import DATA_PATH
from adeft_app.filenames import escape_filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use adeft to find longforms'
                                     ' associated with shortform')
    parser.add_argument('vars', nargs='*')
    args = parser.parse_args()
    shortforms = args.vars
    agg_name = ':'.join(sorted([escape_filename(shortform)
                                for shortform in shortforms]))
    texts_path = os.path.join(DATA_PATH, 'texts', agg_name,
                              f'{agg_name}_texts.json')
    with open(texts_path, 'r') as f:
        texts = json.load(f)
    texts = texts.values()
    texts = [text for text in texts if text]
    for shortform in shortforms:
        dm = DeftMiner(shortform)
        dm.process_texts(texts)
        longforms = dm.get_longforms()
        escaped_shortform = escape_filename(shortform)
        out_path = os.path.join(DATA_PATH, 'longforms',
                                f'{escaped_shortform}_longforms.json')
        with open(out_path, 'w') as f:
            json.dump(longforms, f)
        out_path = os.path.join(DATA_PATH, 'longforms',
                                f'{escaped_shortform}_top.json')
        with open(out_path, 'w') as f:
            json.dump(dm.top(100), f)
