# 76b — repair the phase-74 cached-append gap

Revision: `hotpath-v10-20260914`. Task: `76-02`.

Measure the missing 256-token cached-prefix reuse at the established coordinate
without counting cached tokens as new prefill or conflating it with append size.
