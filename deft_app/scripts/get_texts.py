import os
import pickle
import argparse

from deft_app.locations import DATA_PATH
from deft_app.content_tools import universal_extract_text
from deft_app.content_tools import get_text_content_from_stmt_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get texts for statements'
                                     ' with agent from a list of shortforms')
    parser.add_argument('vars', nargs='*')
    args = parser.parse_args()
    shortforms = args.vars
    all_stmts = set()
    for shortform in shortforms:
        path = os.path.join(DATA_PATH, 'statements',
                            f'{shortform}_statements.pkl')
        with open(path, 'rb') as f:
            stmts = pickle.load(f)
        all_stmts.update(stmts)
        stmt_dict, text_dict = get_text_content_from_stmt_ids(all_stmts)
    text_dict = {text_ref: universal_extract_text(article,
                                                  contains=shortforms)
                 for text_ref, article in text_dict.items()}
    agg_name = ':'.join(shortforms)
    dir_path = os.path.join(DATA_PATH, 'texts', agg_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, f'{agg_name}_texts.pkl'), 'wb') as f:
        pickle.dump(text_dict, f)
    with open(os.path.join(dir_path, f'{agg_name}_text_map.pkl'), 'wb') as f:
        pickle.dump(stmt_dict, f)
