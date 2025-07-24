from datetime import datetime, timezone


def sql_format(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def sql_datetime_format(dt: datetime | None = None) -> str:
    if dt:
        return sql_format(dt)

    return sql_format(datetime.now(timezone.utc))
