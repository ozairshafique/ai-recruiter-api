from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)) -> None:
    configured_key = get_settings().delete_api_key
    if configured_key is None:
        return
    if not api_key or api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
