FROM --platform=linux/amd64 ghcr.io/multimodal-art-projection/nl2repobench/math-verify:1.0

RUN python -m pip install --no-cache-dir 'openai==2.54.0'

RUN mv /workspace /benchmark \
    && chmod -R 700 /benchmark \
    && useradd --create-home --shell /bin/bash agent \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /installed-agent /logs/agent /logs/verifier /logs/artifacts \
    && chown -R agent:agent /app /installed-agent
WORKDIR /app
