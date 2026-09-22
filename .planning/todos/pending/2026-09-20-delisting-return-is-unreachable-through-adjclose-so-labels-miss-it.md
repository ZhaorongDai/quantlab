---
created: "2026-09-20T00:00:00.000Z"
title: The delisting return is unreachable through adjClose, so labels never see the final move
area: data / label
severity: major
status: pending
triggers: >
  any label or factor that derives returns from `adjClose` alone (today: `label/fret.py:Return`,
  and every Alpha101/Alpha158 factor reading `Input("adjClose")`) on a CRSP panel containing a
  modern `DA`-shape delisting. Silent — the panel is well formed and nothing raises.
files:
  - quantlab/dataset/crsp/__init__.py (:550-557 — the recorded deliberate choice)
  - quantlab/label/fret.py (Return — reads Input("adjClose") and nothing else)
  - example/wrds_crsp.md (:78-81 — the survivorship-bias guarantee already narrowed to close/ret/is_delisting)
---

# The delisting return is unreachable through `adjClose`

Found by the Phase 03.10 re-verification (`03.10-VERIFICATION.md`, filed as advisory rather than a
gap — it is a design decision, not a defect).

## Problem

For the modern CRSP `DA` delisting shape, the delisting row legitimately has **no price**: CRSP
writes `dlyprc` as a no-price sentinel and `dlyclose` as NULL, because no trade happened that day.
Phase 03.10 correctly stopped treating that sentinel as a $0.00 close (GAP-A/GAP-B), so `close` is
NaN on that row — and therefore `adjClose` is NaN too.

The consequence: **no `adjClose` ratio spans the delisting day.** The adjusted *level* is now
correct, but the final move — the liquidation gain or loss — is invisible to anything that derives
returns from `adjClose`. `label/fret.py:Return` reads `Input("adjClose")` and nothing else, so the
label simply never sees it.

The return itself is not lost from the panel: `ret` carries it (verified — WRK's delisting row keeps
`ret = -0.00563` intact), and `is_delisting` marks the row. It is only unreachable *through the
`adj*` columns*, which is what every current consumer reads.

## Why this matters more than the observed magnitude suggests

In the tier measured so far the effect is tiny: 5 `DA` rows, WRK's final move is −0.563%. But the
magnitude is not the point — the *shape* is. A near-total liquidation loss (a −90% or −100% final
move) is exactly the observation a survivorship-bias-aware pipeline must not drop, and it is exactly
the one that would be dropped here. The same class of problem as `WR-05`, arriving by a different
route: WR-05 discards the row, this keeps the row but makes the label blind to it.

## Two ways out (decide, don't patch silently)

1. **Require consumers to read `ret`.** Cheapest. `ret` already carries the delisting return.
   Cost: every label/factor that currently reads `adjClose` for returns needs auditing and changing,
   and the "adjusted price is the one true series" convention weakens.
2. **Publish a total-return level** that carries the liquidation return — an `adjClose`-like series
   whose final step embeds `ret` on the delisting row, so a ratio does span the delisting day.
   Cost: a new column and a new invariant to maintain; must not disturb the existing `adjClose`
   contract that Phase 03.10 just pinned with hand-checkable arithmetic.

## Notes

- This is **not** blocked by, and does not block, Phase 03.11 (the PERMNO symbol-axis migration).
  03.11 is a refactor of the identity axis; this is a correctness question about which column
  carries the return. Keeping them separate was a deliberate choice.
- `example/wrds_crsp.md:78-81` has already narrowed the documented survivorship-bias guarantee to
  `close` / `ret` / `is_delisting`, so the documentation is honest today — but no consumer has been
  changed to match it.
