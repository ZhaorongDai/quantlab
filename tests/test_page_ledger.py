"""Page-level resume across a batched, paginated vendor fetch (03.2 SC-3, D-05).

A `us_all` backfill issues thousands of multi-symbol batch requests, and Alpaca
returns each batch as an opaque-token-chained sequence of pages sorted by
symbol first, then by bar timestamp. Restarting an interrupted run from the
start of a batch re-burns every page already paid for; restarting from the
wrong page silently drops the symbols in between. `base/pageledger.py:PageLedger`
records the furthest position reached per batch so a resumed run continues
mid-batch, and records enough alongside the verbatim token to re-derive a
resume point WITHOUT one, because Alpaca publishes no statement about token
lifetime either way (CONTEXT.md D-03).

The token is a position, not a lease: Alpaca's own published example decodes to
`SYMBOL|TIMEFRAME|TIMESTAMP`. That encoding is undocumented, so the ledger
records tokens verbatim and never re-derives them.

Every test here is offline. The page sequence comes from `mock_alpaca_client`,
so nothing sleeps for real, makes a network call, requires `APCA_API_KEY_ID` /
`APCA_API_SECRET_KEY`, or touches any real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-tests; 03.2-02
Task 3 fills in the resume / idempotency / fingerprint tests. It is
deliberately NOT an empty placeholder: a pytest file with zero collected tests
exits 5 ("no tests ran"), which a later task's automated command reads as green.
"""


def test_mock_alpaca_client_default_pages_support_a_midway_resume(
    mock_alpaca_client,
):
    """Fixture self-test: the default page sequence is a real multi-page chain.

    Every resume test depends on exactly this precondition -- at least three
    pages, a non-`None` token on every page but the last, and `None` on the
    last. A two-page or single-page default would make "resumed from page 2"
    indistinguishable from "restarted the batch", and a sequence whose last
    page still carried a token would make batch completion untestable.
    """
    pages = mock_alpaca_client.pages

    assert len(pages) >= 3, (
        f"the default sequence must have >= 3 pages so a resume point can sit "
        f"strictly between the first and the last; got {len(pages)}"
    )

    assert pages[-1]["next_page_token"] is None, (
        "the last page must terminate the chain with next_page_token=None -- "
        "batch completion is what gates the D-04 no-data markers"
    )

    for index, page in enumerate(pages[:-1]):
        assert page["next_page_token"] is not None, (
            f"page {index} of {len(pages)} carries next_page_token=None but is "
            f"not the last page; the chain would terminate early"
        )


def test_mock_alpaca_client_default_pages_reproduce_the_pitfall_4_trap(
    mock_alpaca_client,
):
    """Fixture self-test: a requested symbol is absent from an early page and
    present on a later one -- RESEARCH Pitfall 4, made reproducible.

    Alpaca sorts symbol-major, so "absent from this page" is not "absent from
    the batch". A no-data marker computed per page would stamp the symbols that
    simply have not been reached yet. This fixture sequence is what makes that
    bug reproducible offline; without it a per-page implementation would pass
    every test.
    """
    pages = mock_alpaca_client.pages

    symbols_by_page = [set(page["bars"]) for page in pages]
    all_symbols = set().union(*symbols_by_page)

    late_arrivals = {
        symbol
        for symbol in all_symbols
        if symbol not in symbols_by_page[0]
        and any(symbol in later for later in symbols_by_page[1:])
    }

    assert late_arrivals, (
        f"no symbol is absent from page 0 and present later; the default "
        f"sequence does not reproduce Pitfall 4. Pages held {symbols_by_page}"
    )

    # And the first page really is symbol-major: one symbol, not a slice of all.
    assert len(symbols_by_page[0]) == 1, (
        f"page 0 must hold exactly one symbol to mirror the vendor's documented "
        f"symbol-major ordering; got {sorted(symbols_by_page[0])}"
    )


def test_mock_alpaca_client_records_calls_without_consuming_a_failed_page(
    mock_alpaca_client,
):
    """Fixture self-test for the `raise_on` seam every resume test drives.

    A failing call must be RECORDED (so its `page_token` is assertable) and must
    NOT advance the page queue -- the request never succeeded, so the position
    is unchanged. Getting this backwards would make a resume test pass while the
    implementation silently skipped a page.
    """
    client = mock_alpaca_client
    pages_before = len(client.pages)
    boom = RuntimeError("simulated transport failure")
    client.raise_on = {0: boom}

    instance = client()
    try:
        instance.get_page("/v2/stocks/bars", {"symbols": "AAPL,MSFT"})
    except RuntimeError as exc:
        assert exc is boom
    else:  # pragma: no cover - the fixture is broken if we get here
        raise AssertionError("raise_on did not raise on call index 0")

    assert len(client.calls) == 1
    assert client.calls[0]["path"] == "/v2/stocks/bars"
    assert client.calls[0]["symbols"] == "AAPL,MSFT"
    assert len(client.pages) == pages_before, (
        "a failed call must not consume a page -- the queue position is the "
        "resume position"
    )
