# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**alchemiscale** is a high-throughput alchemical free energy execution system for HPC, cloud, bare metal, and Folding@Home. It's a Python 3.11+ service-oriented system using Neo4j as a graph database state store and AWS S3 for object storage.

## Build & Development Commands

```bash
# Install (editable, no-deps since conda handles deps)
pip install --no-deps -e .

# Environment setup via conda/mamba
micromamba create -f devtools/conda-envs/test.yml
micromamba activate alchemiscale-test

# Run all tests (parallel)
pytest -n auto -v --cov=alchemiscale alchemiscale/tests

# Run a single test file
pytest alchemiscale/tests/unit/test_models.py

# Run a single test by name
pytest -k "test_name_pattern" alchemiscale/tests

# Unit tests only
pytest alchemiscale/tests/unit

# Integration tests only (requires Docker for Neo4j via grolt)
pytest alchemiscale/tests/integration

# Formatting
black alchemiscale
black --check --diff alchemiscale   # check only (CI uses this)

# Type checking
mypy alchemiscale
```

## Architecture

The system has four main services communicating through Neo4j and S3:

- **AlchemiscaleAPI** (`interface/api.py`): User-facing FastAPI REST API for submitting networks, actioning tasks, retrieving results. Deployed with Gunicorn + Uvicorn.
- **AlchemiscaleComputeAPI** (`compute/api.py`): Compute-facing FastAPI REST API for task claiming and result submission.
- **SynchronousComputeService** (`compute/service.py`): Runs on HPC/GPU nodes, claims tasks, executes GUFE ProtocolDAGs, submits results.
- **StrategistService** (`strategist/service.py`): Automated strategy execution service using the `stratocaster` framework.

### Key Modules

- `models.py`: Core data models — `Scope` (org/campaign/project hierarchy) and `ScopedKey` (globally unique identifier within a scope).
- `storage/statestore.py`: `Neo4jStore` — the largest module, contains all Neo4j Cypher query logic for state management. This is the source of truth for the system.
- `storage/objectstore.py`: `S3ObjectStore` — stores serialized ProtocolDAG results.
- `base/api.py`: Shared FastAPI router logic, JWT auth, scope validation, response handling. Both APIs inherit from this.
- `base/client.py`: Shared HTTP client with retry logic. Both `AlchemiscaleClient` and `AlchemiscaleComputeClient` inherit from this.
- `security/auth.py`: JWT token creation/validation, password hashing with bcrypt.
- `settings.py`: Pydantic `BaseSettings` classes that auto-populate from environment variables (Neo4j, S3, JWT, API settings).
- `compression.py`: Zstandard compression/decompression for GUFE objects.
- `cli.py`: Click-based CLI entry point (`alchemiscale` command) for starting APIs, compute services, database management, and identity management.

### Data Flow

1. Users interact via `AlchemiscaleClient` → `AlchemiscaleAPI`
2. Tasks are created and stored in Neo4j with scope-based access control
3. Compute services claim tasks via `AlchemiscaleComputeAPI`
4. Results (ProtocolDAGResults) are stored in S3, references in Neo4j
5. Users retrieve results through the client

### Key Patterns

- **Settings from environment**: All `*Settings` classes use `pydantic-settings` to auto-populate from env vars (case-insensitive).
- **Scope-based authorization**: All entities are organized by `Scope` (org/campaign/project). JWT tokens encode permitted scopes.
- **Frozen Pydantic models**: Core models use `frozen=True` for immutability.
- **Module docstrings**: Follow the pattern `:mod:\`alchemiscale.module_name\` --- description`.
- **GUFE integration**: Core chemistry types (`AlchemicalNetwork`, `Transformation`, `ChemicalSystem`, `Protocol`) come from the `gufe` library and are stored/retrieved via their tokenization system (`GufeKey`, `GufeTokenizable`).

## Testing

Integration tests require Docker (for Neo4j via `grolt`) and use `moto` for AWS S3 mocking. The `conftest.py` in `tests/integration/` handles port management for pytest-xdist parallel workers. Test environments are defined in `devtools/conda-envs/test.yml`.

## CLI Entry Point

The `alchemiscale` CLI (defined in `cli.py`) provides subcommands: `api`, `compute synchronous`, `strategist`, `database init/check/reset/migrate`, and `identity add/list/remove/add_scope`.
