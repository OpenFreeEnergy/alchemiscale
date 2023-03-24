from alchemiscale.settings import ComputeAPISettings


def get_compute_settings_override():
    # settings overrides for test suite
    return ComputeAPISettings(
        NEO4J_USER="neo4j",
        NEO4J_PASS="password",
        NEO4J_URL="bolt://localhost:7687",
        ALCHEMISCALE_COMPUTE_API_HOST="127.0.0.1",
        ALCHEMISCALE_COMPUTE_API_PORT=8000,
        ALCHEMISCALE_COMPUTE_API_REGISTRATION_EXPIRE_SECONDS=1800,
        JWT_SECRET_KEY="98d11ba9ca329a4e5a6626faeffc6a9b9fb04e2745cff030f7d6793751bb8245",
        JWT_EXPIRE_SECONDS=10,
        AWS_ACCESS_KEY_ID="test-key-id",
        AWS_SECRET_ACCESS_KEY="test-key",
        AWS_SESSION_TOKEN="test-session-token",
        AWS_S3_BUCKET="test-bucket",
        AWS_S3_PREFIX="test-prefix",
        AWS_DEFAULT_REGION="us-east-1",
    )
