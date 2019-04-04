import os
import re
import json
import argparse


from deft_app.locations import DATA_PATH
from deft_app.content_tools import get_stmts_with_agent_text_like


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get statements with agent'
                                     'text matching a pattern')
    parser.add_argument('pattern')
    parser.add_argument('keep', nargs='?')

    args = parser.parse_args()
    pattern = args.pattern
    keep = args.keep
    if not keep:
        keep = ''
    keep = re.compile(keep)
    stmt_dict = get_stmts_with_agent_text_like(pattern,
                                               filter_genes=True)
    for shortform, stmts in stmt_dict.items():
        if re.match(keep, shortform):
            with open(os.path.join(DATA_PATH, 'statements',
                                   f'{shortform}_statements.json'), 'w') as f:
                json.dump(stmts, f)
