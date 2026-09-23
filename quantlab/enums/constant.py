"""Shared constants that are not tied to one data vendor or market."""

from dataclasses import dataclass


@dataclass
class Date:
    """Sentinel date bounds meaning "no lower bound" and "no upper bound".

    A factor or dataset whose config leaves ``start_date`` or ``end_date``
    unset is given these values, so a date filter always has two concrete
    endpoints to compare against.
    """

    START_DATE = "1900-01-01"
    END_DATE = "2100-01-01"
