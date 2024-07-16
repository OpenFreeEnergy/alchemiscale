try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        validator,
        root_validator,
        BaseSettings,
        ValidationError,
    )
except ImportError:
    from pydantic import (
        BaseModel,
        Field,
        validator,
        root_validator,
        BaseSettings,
        ValidationError,
    )
