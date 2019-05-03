import json
from nltk.corpus import words

from indra_db.util.content_scripts import get_stmts_with_agent_text_in

bogosity = get_stmts_with_agent_text_in(words.words(), filter_genes=True)
with open(f'../data/genes_with_english_names.json', 'w') as f:
    json.dump(bogosity, f)
