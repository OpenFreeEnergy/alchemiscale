from alchemiscale.settings import APISettings


def get_user_settings_override():
    # settings overrides for test suite
    return APISettings(
        NEO4J_USER="neo4j",
        NEO4J_PASS="password",
        NEO4J_URL="bolt://localhost:7687",
        ALCHEMISCALE_API_HOST="127.0.0.1",
        ALCHEMISCALE_API_PORT=8000,
        JWT_SECRET_KEY="3f072449f5f496d30c0e46e6bc116ba27937a1482c3a4e41195be899a299c7e4",
        JWT_EXPIRE_SECONDS=3,
        AWS_ACCESS_KEY_ID="test-key-id",
        AWS_SECRET_ACCESS_KEY="test-key",
        AWS_SESSION_TOKEN="test-session-token",
        AWS_S3_BUCKET="test-bucket",
        AWS_S3_PREFIX="test-prefix",
        AWS_DEFAULT_REGION="us-east-1",
    )
