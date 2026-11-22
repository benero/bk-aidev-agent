FROM node:24-bookworm AS engine
ARG OPENCLAW_VERSION=2026.11.6
ARG NPM_CONFIG_REGISTRY=https://registry.npmjs.org
RUN npm install --global openclaw@${OPENCLAW_VERSION} && npm cache clean --force

FROM python:3.11-slim-bookworm
COPY --from=engine /usr/local/bin/node /usr/local/bin/node
COPY --from=engine /usr/local/lib/node_modules/openclaw /usr/local/lib/node_modules/openclaw
RUN ln -s /usr/local/lib/node_modules/openclaw/openclaw.mjs /usr/local/bin/openclaw
WORKDIR /opt/adapter
COPY pyproject.toml readme.md ./
COPY aidev_agent_adapter/ ./aidev_agent_adapter/
RUN pip install --no-cache-dir .
COPY examples/openclaw/demo.py /opt/adapter/demo.py
RUN useradd --create-home adapter && install -d -o adapter -g adapter -m 700 /var/lib/aidev-adapter
USER adapter
ENTRYPOINT ["python","/opt/adapter/demo.py"]
CMD ["--config","/run/aidev/config.json"]
