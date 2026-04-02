from dataclasses import dataclass, field

@dataclass
class CapnostreamContainer:
    path: str | None = None

@dataclass
class MotionContainer:
    paths: set[str] = field(default_factory=set)