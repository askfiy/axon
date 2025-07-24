from enum import StrEnum


def enum_values(enum_class: type[StrEnum]) -> list[str]:
    return [e.value for e in enum_class]
