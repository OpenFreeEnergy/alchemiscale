import gunicorn.app.base


class ApiApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, workers, bind, options=None):
        self.app = app
        self.workers = workers
        self.bind = bind
        self.options = options or {}
        super().__init__()

    @classmethod
    def from_parameters(cls, app, workers, host, port, options=None):
        return cls(app, workers, bind=f"{host}:{port}", options=options)

    def load(self):
        return self.app

    def load_config(self):
        self.cfg.set("workers", self.workers)
        self.cfg.set("bind", self.bind)
        self.cfg.set("worker_class", "uvicorn.workers.UvicornWorker")

        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }

        for key, value in config.items():
            self.cfg.set(key.lower(), value)
