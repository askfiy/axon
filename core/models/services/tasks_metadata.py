from pydantic import field_serializer

from ..model import BaseModel


# Tips: 我们的 keywords 入站规则是 list[str]. 但是 db 中是 str.
# 若要返回给外部。 则需要保持设计的一致性将其反序列化为 list[str].
# 目前 meta_info 不会出站. 故暂时搁置.
class TaskMetaDataRequestModel(BaseModel):
    owner: str
    owner_timezone: str
    keywords: list[str]
    original_user_input: str
    planning: str
    description: str
    accept_criteria: str

    @field_serializer("keywords")
    def _validator_keywords(self, keywords: list[str]) -> str:
        return ",".join(keywords)
