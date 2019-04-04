import os
import pickle
import argparse

from indra_db.util.content_scripts import get_stmts_with_agent_text_like

from deft_app.locations import DATA_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get statements with agent'
                                     'text matching a pattern')
    parser.add_argument('pattern')
    parser.add_argument('outfile')

    args = parser.parse_args()
    pattern = args.pattern
    outfile = args.outfile
    stmt_dict = get_stmts_with_agent_text_like(pattern,
                                               filter_genes=True)
    with open(os.path.join(DATA_PATH, 'statements', outfile), 'wb') as f:
        pickle.dump(stmt_dict, f)
