"""
API 依赖项
"""
from typing import Annotated

from fastapi import Depends, HTTPException, Path, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.base import get_db
from app.models.user import User
from app.utils.security import decode_access_token


security = HTTPBearer()

# 路径里的会话 id：不是 UUID 时直接返回 422，不让它走到数据库报错。
ConversationId = Annotated[
    str, Path(pattern=r"^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$")
]


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    获取当前登录用户（依赖注入）

    用法：
        @app.get("/me")
        async def get_me(user: User = Depends(get_current_user)):
            return user
    """

    token = credentials.credentials

    # 解码 JWT
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期"
        )

    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌格式错误"
        )

    # 查询用户
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )

    return user
