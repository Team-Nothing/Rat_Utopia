import importlib
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


description = """
Rat Recorder API helps you do awesome stuff. 🚀
"""


with open("service_config.json", "r") as f:
    config = json.load(f)


def app_lifespan(app: FastAPI):
    try:
        for file_name in os.listdir("routes"):
            if file_name.endswith(".py") and file_name != "__init__.py":
                module = importlib.import_module(f"routes.{file_name[:-3]}")
                if hasattr(module, "startup"):
                    getattr(module, "startup")(config)

        yield

    finally:
        for file_name in os.listdir("routes"):
            if file_name.endswith(".py") and file_name != "__init__.py":
                module = importlib.import_module(f"routes.{file_name[:-3]}")
                if hasattr(module, "shutdown"):
                    getattr(module, "shutdown")()


app = FastAPI(
    title="Rat Recorder API",
    description=description,
    summary="API",
    version="0.0.1",
    contact={
        "name": "Nothing Chang",
        "url": "https://github.com/I-am-nothing",
        "email": "jdps99119@gmail.com",
    },
    lifespan=app_lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "code": "OK",
        "message": "Medical Record API is running!"
    }


for file_name in os.listdir("routes"):
    if file_name.endswith(".py") and file_name != "__init__.py":
        module = importlib.import_module(f"routes.{file_name[:-3]}")
        if hasattr(module, "setup"):
            getattr(module, "setup")(app, config)