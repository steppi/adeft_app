import os
from joblib import Memory
from flask import current_app

from indra.sources import trips
from indra.databases import hgnc_client

trips_cache = os.path.join('.cache')
memory = Memory(trips_cache, verbose=0)


def _trips_ground(text):
    """Attempt to ground an agent text with trips

    Searches text for trips groundings. For a grounding to be returned
    the text for any extracted agent must match the input text in a case
    insensitive way. A priority is given for different namespaces.
    HGNC > FPLX > UP (not cellular location) > GO > CHEBI > MESH >
    UP (cellular location). Only these name spaces are considered.

    Parameters
    ----------
    text : str
        An agent text

    Returns
    -------
    name : str
        Agent name if a trips grounding was found, otherwise None
    grounding : str
        Grounding of the form <name_space>:<id> as contained in an
        Indra agent's db_refs
    """
    try:
        tp = trips.process_text(text, service_endpoint='drum-dev')
    except Exception:
        return None, None
    agents = tp.get_agents()
    # filter to agents with text matching input text
    proper_agents = [agent for agent in agents if
                     'TEXT' in agent.db_refs
                     and agent.db_refs['TEXT'].lower() ==
                     text.lower()]
    if proper_agents:
        agent = proper_agents[0]
    else:
        return None, None

    name = agent.name

    hgnc_id = agent.db_refs.get('HGNC')
    fplx_id = agent.db_refs.get('FPLX')
    up_id = agent.db_refs.get('UP')
    mesh_id = agent.db_refs.get('MESH')
    chebi_id = agent.db_refs.get('CHEBI')
    go_id = agent.db_refs.get('GO')

    # get grounding according to priorities
    if hgnc_id is not None:
        grounding = 'HGNC:' + hgnc_id
    elif fplx_id is not None:
        grounding = 'FPLX:' + fplx_id
    elif up_id is not None and not up_id.startswith('SL-'):
        grounding = 'UP:' + up_id
    elif go_id is not None:
        grounding = 'GO:' + go_id
    elif chebi_id is not None:
        grounding = chebi_id
    elif mesh_id is not None:
        grounding = 'MESH:' + mesh_id
    elif up_id is not None:
        grounding = 'UP:' + up_id
    else:
        grounding = None

    return name, grounding


trips_ground = memory.cache(_trips_ground)
