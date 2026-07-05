from pydantic import BaseModel, ConfigDict


class TextRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "People trapped under collapsed building. Send rescue immediately!"
            }
        }
    )

    text: str
