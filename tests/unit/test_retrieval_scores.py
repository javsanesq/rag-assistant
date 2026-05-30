from rag_assistant_api.adapters.vector_store import _lexical_score, _tokenize


def test_lexical_score_counts_query_term_overlap():
    query_terms = _tokenize("annual refund policy")
    chunk_terms = _tokenize("annual plans have a refund window")
    assert _lexical_score(query_terms, chunk_terms) == 2 / 3
