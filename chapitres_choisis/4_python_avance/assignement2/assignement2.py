from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal


@dataclass
class User:
    user_nickname: str
    user_id: int
    user_email: str
    user_second_email: Optional[str]
    user_level: Literal["Normal", "Moderator", "Admin"]
    user_adult: bool
    user_last_login: datetime
    user_email_verified: bool

    def __init__(self, nickname: str, id: int, email: str, email2: Optional[str],
                 level: Literal["Normal", "Moderator", "Admin"], adult: bool) -> None:
        self.user_nickname = nickname
        self.user_id = id
        self.user_email = email
        self.user_second_email = email2
        self.user_level = level
        self.user_adult = adult
        self.user_email_verified = False
        self.user_last_login = datetime.now()

    def login(self) -> None:
        self.user_last_login = datetime.now()

    def verify(self) -> None:
        self.user_email_verified = True
