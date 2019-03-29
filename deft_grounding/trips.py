import os
from joblib import Memory
from flask import current_app

from indra.sources import trips
from indra.databases import hgnc_client

trips_cache = os.path.join('.cache')
memory = Memory(trips_cache, verbose=0)


def trips_ground(text):
    name = grounding = None
    try:
        tp = trips.process_text(text, service_endpoint='drum-dev')
        terms = tp.tree.findall('TERM')
        if terms:
            term_id = terms[0].attrib['id']
            agent = tp._get_agent_by_id(term_id, None)
            if 'HGNC' in agent.db_refs:
                dbn = 'HGNC'
                dbi = agent.db_refs['HGNC']
                name = hgnc_client.get_hgnc_name(dbi)
                grounding = f'{dbn}:{dbi}'
            elif 'FPLX' in agent.db_refs:
                dbn = 'FPLX'
                dbi = agent.db_refs['FPLX']
                name = dbi
                grounding = f'{dbn}:{dbi}'
    except Exception:
        pass
    return name, grounding


trips_ground = memory.cache(trips_ground)
