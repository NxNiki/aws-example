from datetime import datetime, timedelta
from typing import Literal

AggCol = Literal["activity_date", "activity_week", "activity_month"]


def effective_start_date(stats_agg_col: AggCol, start_date: str) -> str:
    """
    Truncate start_date to the start of the aggregation period (day/week/month)
    so incremental jobs always fetch full periods and avoid losing data when
    lookback is shorter than a full week or month.
    """
    dt = datetime.strptime(start_date, "%Y-%m-%d").date()

    if stats_agg_col == "activity_month":
        dt = dt.replace(day=1)
    elif stats_agg_col == "activity_week":
        # Monday as start of week, aligned with DATE_TRUNC('week', ...) usage
        dt = dt - timedelta(days=dt.weekday())

    return dt.strftime("%Y-%m-%d")
