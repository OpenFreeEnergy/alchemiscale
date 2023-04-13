import gunicorn.app.base

class ApiApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, workers, bind):
        self.app = app
        self.workers = workers
        self.bind = bind
        super().__init__()

    @classmethod
    def from_parameters(cls, app, workers, host, port):
        return cls(app, workers, bind=f"{host}:{port}")

    def load(self):
        return self.app

    def load_config(self):
        self.cfg.set("workers", self.workers)
        self.cfg.set("bind", self.bind)
        self.cfg.set("worker_class", "uvicorn.workers.UvicornWorker")

