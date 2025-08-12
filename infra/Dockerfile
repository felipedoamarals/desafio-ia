FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

COPY ../pyproject.toml ../poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --without dev

COPY ../app ./app

EXPOSE 8888
CMD ["poetry","run","jupyter","lab","--ip=0.0.0.0","--port=8888","--NotebookApp.notebook_dir=app/notebook","--no-browser","--allow-root"]
