FROM mambaorg/micromamba:1.1.0

COPY --chown=$MAMBA_USER:$MAMBA_USER devtools/conda-envs/test.yml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER fah_alchemy /tmp/fah_alchemy
COPY --chown=$MAMBA_USER:$MAMBA_USER LICENSE MANIFEST.in README.md setup.cfg setup.py versioneer.py /tmp/
RUN micromamba install -y -n base git -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -m pip install --no-deps .
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "fah-alchemy"]
