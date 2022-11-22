from fah_alchemy.settings import APISettings


def get_user_settings_override():
    # settings overrides for test suite
    return APISettings(
        NEO4J_USER="neo4j",
        NEO4J_PASS="password",
        NEO4J_URL="bolt://localhost:7687",
        FA_API_HOST="127.0.0.1",
        FA_API_PORT=8000,
        JWT_SECRET_KEY="3f072449f5f496d30c0e46e6bc116ba27937a1482c3a4e41195be899a299c7e4",
        JWT_EXPIRE_SECONDS=3,
    )
