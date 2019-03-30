import time
import logging
from collections import defaultdict

import indra_db.util as dbu
from indra.literature import pubmed_client, pmc_client, elsevier_client

logger = logging.getLogger(__name__)

# the elsevier_client will log messages that it is safe to ignore
el = logging.getLogger('indra.literature.elsevier_client')
el.setLevel(logging.WARNING)


def get_plaintexts(text_content, contains=None):
    """Returns a corpus of plaintexts given text content from different sources

    Converts xml files into plaintext, leaves abstracts as they are.

    Parameters
    ----------
    sources : list of str
        lists of text content. each item should either be a plaintext, an
        an NLM xml or an Elsevier xml

    Returns
    -------
    plaintexts : list of str
        list of plaintexts for input list of xml strings
    """
    return [universal_extract_text(article, contains)
            for article in text_content]


def universal_extract_text(xml, contains=None):
    """Extract plaintext from xml

    First try to parse the xml as if it came from elsevier. if we do not
    have valid elsevier xml this will throw an exception. the text extraction
    function in the pmc client may not throw an exception when parsing elsevier
    xml, silently processing the xml incorrectly

    Parameters
    ----------
    xml : str
       Either an NLM xml, Elsevier xml or plaintext

    Returns
    -------
    plaintext : str
        for NLM or Elsevier xml as input, this is the extracted plaintext
        otherwise the input is returned unchanged
    """
    try:
        plaintext = elsevier_client.extract_text(xml, contains)
    except Exception:
        plaintext = None
    if plaintext is None:
        try:
            plaintext = pmc_client.extract_text(xml, contains)
        except Exception:
            plaintext = xml
    return plaintext


def get_statements_with_agent_text_like(pattern):
    """Get statement ids with agent with rawtext matching pattern


    Parameters
    ----------
    pattern : str
        a pattern understood by sqlalchemy's like operator.
        For example '__' for two letter agents

    Returns
    -------
    dict
        dict mapping agent texts matching the input pattern to lists of
        ids for statements with at least one agent with raw text matching
        the pattern.
    """
    db = dbu.get_primary_db()
    # get all stmts with at least one hgnc grounded agent
    hgnc_stmts = db.select_all(db.RawAgents.stmt_id,
                               db.RawAgents.db_name == 'HGNC',
                               db.RawAgents.stmt_id.isnot(None))
    hgnc_stmts = set(stmt_id[0] for stmt_id in hgnc_stmts)
    text_dict = db.select_all([db.RawAgents.stmt_id,
                               db.RawAgents.db_id],
                              db.RawAgents.db_name == 'TEXT',
                              db.RawAgents.db_id.like(pattern),
                              db.RawAgents.stmt_id.isnot(None))
    hgnc_rawtexts = set()
    for stmt_id, db_id in text_dict:
        if stmt_id not in hgnc_stmts:
            continue
        hgnc_rawtexts.add(db_id)

    result_dict = defaultdict(list)
    for stmt_id, db_id in text_dict:
        if db_id in hgnc_rawtexts:
            result_dict[db_id].append(stmt_id)
    return dict(result_dict)


def get_text_content_from_stmt_ids(stmt_ids):
    """Get text content for statements from a list of ids

    Gets the fulltext if it is available, even if the statement came from an
    abstract.

    Parameters
    ----------
    stmt_ids : list of str

    Returns
    -------
    dict of str: str
        dictionary mapping statement ids to text content. Uses fulltext
        if one is available, falls back upon using the abstract.
        A statement id will map to None if no text content is available.
    """
    db = dbu.get_primary_db()
    text_refs = db.select_all([db.RawStatements.id, db.TextRef.id],
                              db.RawStatements.id.in_(stmt_ids),
                              *db.link(db.RawStatements, db.TextRef))
    text_refs = dict(text_refs)
    texts = db.select_all([db.TextContent.text_ref_id,
                           db.TextContent.content,
                           db.TextContent.text_type],
                          db.TextContent.text_ref_id.in_(text_refs.values()))
    fulltexts = {text_id: dbu.unpack(text)
                 for text_id, text, text_type in texts
                 if text_type == 'fulltext'}
    abstracts = {text_id: dbu.unpack(text)
                 for text_id, text, text_type in texts
                 if text_type == 'abstract'}
    result = {}
    for stmt_id in stmt_ids:
        # first check if we have text content for this statement
        try:
            text_ref = text_refs[stmt_id]
        except KeyError:
            # if not, set fulltext to None
            result[stmt_id] = None
            continue
        fulltext = fulltexts.get(text_ref)
        abstract = abstracts.get(text_ref)
        # use the fulltext if we have one
        if fulltext is not None:
            # if so, the text content is xml and will need to be processed
            result[stmt_id] = fulltext
        # otherwise use the abstract
        elif abstract is not None:
            result[stmt_id] = abstract
        # if we have neither, set result to None
        else:
            result[stmt_id] = None
    return result


def get_text_content_for_gene(hgnc_name):
    """Get articles that have been annotated to contain gene in entrez

    Parameters
    ----------
    hgnc_name : str
       HGNC name for gene

    Returns
    -------
    text_content : list of str
        xmls of fulltext if available otherwise abstracts for all articles
        that haven been annotated in entrez to contain the given gene
    """
    pmids = pubmed_client.get_ids_for_gene(hgnc_name)
    return get_text_content_for_pmids(pmids)


def get_text_content_for_pmids(pmids):
    """Get text content for articles given a list of their pmids

    Parameters
    ----------
    pmids : list of str

    Returns
    -------
    text_content : list of str
    """
    pmc_pmids = set(pmc_client.filter_pmids(pmids, source_type='fulltext'))

    pmc_ids = []
    for pmid in pmc_pmids:
        pmc_id = pmc_client.id_lookup(pmid, idtype='pmid')['pmcid']
        if pmc_id:
            pmc_ids.append(pmc_id)
        else:
            pmc_pmids.discard(pmid)

    pmc_xmls = []
    failed = set()
    for pmc_id in pmc_ids:
        if pmc_id is not None:
            pmc_xmls.append(pmc_client.get_xml(pmc_id))
        else:
            failed.append(pmid)
        time.sleep(0.5)

    remaining_pmids = set(pmids) - pmc_pmids | failed
    abstracts = []
    for pmid in remaining_pmids:
        abstract = pubmed_client.get_abstract(pmid)
        abstracts.append(abstract)
        time.sleep(0.5)

    return [text_content for source in (pmc_xmls, abstracts)
            for text_content in source if text_content is not None]
