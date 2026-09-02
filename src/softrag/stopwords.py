"""Stopwords for keyword retrieval.

These words are dropped when building an FTS5 query. The reason is specific and
measurable rather than traditional: an OR over every token in *"when can we ship
code to customers"* matches whichever documents happen to contain *can*, *we*
and *to*, and rank fusion then promotes that noise above correct dense hits. On
a labelled set this made hybrid search worse than vector search alone.

BM25 cannot rescue it, because BM25 only reweights documents that already
matched -- it never gets to reject a candidate set assembled from stopwords.

Document-frequency filtering handles the same problem on a large corpus, and the
store applies both. A fixed list is what covers the small-corpus case, where
every term looks rare and frequencies say nothing.

English and Portuguese are covered because they are the languages softrag is
developed and tested against. The set is not exhaustive and does not need to be:
missing a stopword costs a little precision, never correctness. Extend or
replace it for another language:

    from softrag import stopwords
    stopwords.STOPWORDS = frozenset(my_own_set)
"""

from __future__ import annotations

__all__ = ["STOPWORDS", "ENGLISH", "PORTUGUESE", "is_stopword"]

ENGLISH = frozenset(
    """
    a about above after again against all am an and any are aren't as at
    be because been before being below between both but by
    can cannot can't could couldn't
    did didn't do does doesn't doing don't down during
    each few for from further
    had hadn't has hasn't have haven't having he her here hers herself him
    himself his how
    i if in into is isn't it its itself i've i'm i'll
    just
    let's
    me more most mustn't my myself
    no nor not
    of off on once only or other ought our ours ourselves out over own
    same shan't she should shouldn't so some such
    than that the their theirs them themselves then there these they this
    those through to too
    under until up
    very
    was wasn't we were weren't what when where which while who whom why with
    won't would wouldn't
    you your yours yourself yourselves
    """.split()
)

PORTUGUESE = frozenset(
    """
    a ao aos aquela aquelas aquele aqueles aquilo as ate até
    com como
    da das de dela delas dele deles depois do dos
    e ela elas ele eles em entre era eram essa essas esse esses esta estas
    este estes estou está estão eu
    foi fomos for foram fosse fossem fui
    ha haja hajam havia houve há
    isso isto
    já
    lhe lhes
    mais mas me mesmo meu meus minha minhas muito
    na nas nem no nos nossa nossas nosso nossos num numa não nós
    o os ou
    para pela pelas pelo pelos por porque pra pro
    qual quando que quem
    se seja sejam sem ser seu seus somos sou sua suas são só
    também te tem tenha tenho ter teu teus teve tinha tive tu tua tuas tém
    um uma umas uns
    vez você vocês vos
    à às é
    """.split()
)

#: The set consulted at query time. Reassign it to change behaviour globally.
STOPWORDS: frozenset[str] = ENGLISH | PORTUGUESE


def is_stopword(token: str) -> bool:
    """Whether ``token`` should be dropped from a keyword query.

    Args:
        token: A single word, in any case.

    Returns:
        ``True`` when the token carries no discriminating signal.

    Example:
        >>> is_stopword("The")
        True
        >>> is_stopword("checkpoint")
        False
    """
    return token.lower() in STOPWORDS
