"""部分反代会改写下划线开头的工具名：结构化输出前先换成干净名字的子类。"""

from pydantic import BaseModel

from app.core.llm import _tool_safe


class _Hidden(BaseModel):
    """私有 schema。"""

    city: str


class Public(BaseModel):
    city: str


def test_underscore_schema_gets_a_clean_tool_name_but_stays_the_same_type():
    safe = _tool_safe(_Hidden)

    assert safe.__name__ == "Hidden" and safe.__doc__ == _Hidden.__doc__
    assert isinstance(safe(city="成都"), _Hidden)


def test_other_schemas_pass_through():
    assert _tool_safe(Public) is Public
    assert _tool_safe({"name": "x"}) == {"name": "x"}
