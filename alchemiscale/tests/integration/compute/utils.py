from alchemiscale.settings import ComputeAPISettings


def get_compute_settings_override():
    # settings overrides for test suite
    return ComputeAPISettings(
        ALCHEMISCALE_COMPUTE_API_HOST="127.0.0.1",
        ALCHEMISCALE_COMPUTE_API_PORT=8000,
        ALCHEMISCALE_COMPUTE_API_REGISTRATION_EXPIRE_SECONDS=1800,
        JWT_SECRET_KEY="98d11ba9ca329a4e5a6626faeffc6a9b9fb04e2745cff030f7d6793751bb8245",
        JWT_EXPIRE_SECONDS=10,
    )
