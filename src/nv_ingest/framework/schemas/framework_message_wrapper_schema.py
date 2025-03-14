from pydantic import BaseModel


class MessageWrapper(BaseModel):
    payload: str
