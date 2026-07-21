"""
:mod:`alchemiscale.compute.api` --- compute API components
==========================================================

"""

import json
import datetime
from datetime import timedelta
import random

from fastapi import FastAPI, APIRouter, Body, Depends, HTTPException, Request
from fastapi import status as http_status
from fastapi.middleware.gzip import GZipMiddleware
from gufe.tokenization import JSON_HANDLER, GufeKey
from gufe.protocols import ProtocolDAGResult

from ..base.api import (
    QueryGUFEHandler,
    scope_params,
    get_token_data_depends,
    get_n4js_depends,
    get_s3os_depends,
    base_router,
    get_cred_entity,
    validate_scopes,
    validate_scopes_query,
    minimize_scope_space,
    _check_store_connectivity,
    gufe_to_json,
    GzipRoute,
)
from ..compression import decompress_gufe_zstd
from ..settings import (
    get_base_api_settings,
    get_compute_api_settings,
    ComputeAPISettings,
)
from ..storage.statestore import Neo4jStore
from ..storage.objectstore import S3ObjectStore, protocol_unit_result_location
from ..storage.models import (
    ProtocolDAGResultRef,
    ComputeServiceID,
    ComputeManagerID,
    ComputeManagerRegistration,
    ComputeServiceRegistration,
    ComputeManagerStatus,
)
from ..models import Scope, ScopedKey
from ..security.models import (
    TokenData,
    CredentialedComputeIdentity,
)

app = FastAPI(title="AlchemiscaleComputeAPI")
app.dependency_overrides[get_base_api_settings] = get_compute_api_settings
app.include_router(base_router)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)


def get_cred_compute():
    return CredentialedComputeIdentity


app.dependency_overrides[get_cred_entity] = get_cred_compute


router = APIRouter(
    dependencies=[Depends(get_token_data_depends)],
)
router.route_class = GzipRoute


@app.get("/ping")
def ping():
    return {"api": "AlchemiscaleComputeAPI"}


@router.get("/info")
def info():
    return {"message": "nothing yet"}


@router.get("/check")
def check(
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
):
    # check connectivity of storage components
    # if no exception raised, all good
    _check_store_connectivity(n4js, s3os)


@router.get("/identities/{identity_identifier}/scopes")
def list_scopes(
    *,
    identity_identifier,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
) -> list[str]:
    scopes = n4js.list_scopes(identity_identifier, CredentialedComputeIdentity)
    return [str(scope) for scope in scopes]


@router.post("/computeservice/{compute_service_id}/register")
def register_computeservice(
    compute_service_id,
    *,
    compute_manager_id: str | None = Body(None, embed=True),
    hostname: str | None = Body(None, embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    now = datetime.datetime.now(tz=datetime.UTC)
    if compute_manager_id:
        manager_name = process_compute_manager_id_string(compute_manager_id).name
    else:
        manager_name = None

    csreg = ComputeServiceRegistration(
        identifier=ComputeServiceID(compute_service_id),
        registered=now,
        heartbeat=now,
        failure_times=[],
        manager_name=manager_name,
        hostname=hostname,
    )

    try:
        compute_service_id_ = n4js.register_computeservice(csreg)
    except ValueError as e:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    return compute_service_id_


@router.post("/computeservice/{compute_service_id}/deregister")
def deregister_computeservice(
    compute_service_id,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    compute_service_id_ = n4js.deregister_computeservice(
        ComputeServiceID(compute_service_id)
    )

    return compute_service_id_


@router.post("/computeservice/{compute_service_id}/heartbeat")
def heartbeat_computeservice(
    compute_service_id,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    settings: ComputeAPISettings = Depends(get_base_api_settings),
):
    now = datetime.datetime.now(tz=datetime.UTC)

    # expire any stale registrations, along with their claims
    expire_delta = timedelta(
        seconds=settings.ALCHEMISCALE_COMPUTE_API_REGISTRATION_EXPIRE_SECONDS
    )
    expire_time = now - expire_delta
    n4js.expire_registrations(expire_time)

    compute_service_id_ = n4js.heartbeat_computeservice(compute_service_id, now)

    return compute_service_id_


@router.get("/taskhubs")
def query_taskhubs(
    *,
    return_gufe: bool = False,
    scope: Scope = Depends(scope_params),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    # intersect query scopes with accessible scopes in the token
    query_scopes = validate_scopes_query(scope, token)
    taskhubs_handler = QueryGUFEHandler(return_gufe)

    # query each scope
    # loop might be more removable in the future with a Union like operator on scopes
    for single_query_scope in query_scopes:
        # add new task hubs
        taskhubs_handler.update_results(
            n4js.query_taskhubs(
                scope=single_query_scope, return_gufe=taskhubs_handler.return_gufe
            )
        )

    return taskhubs_handler.format_return()


@router.post("/taskhubs/{taskhub_scoped_key}/claim")
def claim_taskhub_tasks(
    taskhub_scoped_key,
    *,
    compute_service_id: str = Body(),
    count: int = Body(),
    protocols: list[str] | None = Body(None, embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(taskhub_scoped_key)
    validate_scopes(sk.scope, token)

    tasks = n4js.claim_taskhub_tasks(
        taskhub=taskhub_scoped_key,
        compute_service_id=ComputeServiceID(compute_service_id),
        count=count,
        protocols=protocols,
    )

    return [str(t) if t is not None else None for t in tasks]


@router.post("/claim")
def claim_tasks(
    scopes: list[Scope] = Body(),
    scopes_exclude: list[Scope] | None = Body(None, embed=True),
    compute_service_id: str = Body(),
    count: int = Body(),
    protocols: list[str] | None = Body(None, embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    settings: ComputeAPISettings = Depends(get_base_api_settings),
    token: TokenData = Depends(get_token_data_depends),
):
    """Claim ``count`` ``Tasks`` for a given compute service.

    This method returns ``None`` if the compute service request has
    been denied. Otherwise, it returns a list with ``count`` elements.
    These elements are either the string representation of a claimed
    ``Task`` ``ScopedKey``, or ``None``.

    """
    # check if the compute service can claim tasks
    now = datetime.datetime.now(tz=datetime.UTC)
    if not n4js.compute_service_can_claim(
        compute_service_id,
        now - timedelta(seconds=settings.ALCHEMISCALE_COMPUTE_API_FORGIVE_TIME_SECONDS),
        settings.ALCHEMISCALE_COMPUTE_API_MAX_FAILURES,
    ):
        # differs from list[str | None], this shows that the service was
        # actively denied
        return None

    # intersect query scopes with accessible scopes in the token
    scopes_reduced = minimize_scope_space(scopes)
    query_scopes = []
    for scope in scopes_reduced:
        query_scopes.extend(validate_scopes_query(scope, token))

    taskhubs = dict()
    # query each scope for available taskhubs
    # loop might be more removable in the future with a Union like operator on scopes
    for single_query_scope in set(query_scopes):
        taskhubs.update(n4js.query_taskhubs(scope=single_query_scope, return_gufe=True))

    # filter out taskhubs whose scope is covered by any excluded scope
    if scopes_exclude:
        scopes_exclude_reduced = minimize_scope_space(scopes_exclude)

        def is_excluded(taskhub_scope: Scope) -> bool:
            return any(
                excluded.is_superset(taskhub_scope)
                for excluded in scopes_exclude_reduced
            )

        taskhubs = {sk: th for sk, th in taskhubs.items() if not is_excluded(sk.scope)}

    # list of tasks to return
    tasks = []

    if len(taskhubs) == 0:
        # match the pad-with-``None`` behavior of the normal return path below
        return [None] * count

    # claim tasks from taskhubs based on weight; keep going till we hit our
    # total desired task count, or we run out of taskhubs to draw from
    while len(tasks) < count and len(taskhubs) > 0:
        weights = [th.weight for th in taskhubs.values()]

        if sum(weights) == 0:
            break

        # based on weights, choose taskhub to draw from
        taskhub: ScopedKey = random.choices(list(taskhubs.keys()), weights=weights)[0]

        # claim tasks from the taskhub
        claimed_tasks = n4js.claim_taskhub_tasks(
            taskhub,
            compute_service_id=ComputeServiceID(compute_service_id),
            count=(count - len(tasks)),
            protocols=protocols,
        )

        # gather up claimed tasks, if present
        for t in claimed_tasks:
            if t is not None:
                tasks.append(t)

        # remove this taskhub from the options available; repeat
        taskhubs.pop(taskhub)

    return [str(t) for t in tasks] + [None] * (count - len(tasks))


@router.get("/tasks/{task_scoped_key}/transformation")
def get_task_transformation(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(sk.scope, token)

    transformation: ScopedKey

    transformation, _ = n4js.get_task_transformation(
        task=task_scoped_key,
        return_gufe=False,
    )

    return str(transformation)


@router.get("/tasks/{task_scoped_key}/transformation/gufe")
def retrieve_task_transformation(
    task_scoped_key,
    *,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(sk.scope, token)

    transformation_sk, protocoldagresultref_sk = n4js.get_task_transformation(
        task=task_scoped_key, return_gufe=False
    )

    transformation = n4js.get_gufe(transformation_sk)

    if protocoldagresultref_sk:
        protocoldagresultref = n4js.get_gufe(protocoldagresultref_sk)
        pdr_sk = ScopedKey(gufe_key=protocoldagresultref.obj_key, **sk.scope.to_dict())

        # we keep this as a string to avoid useless deserialization/reserialization here
        try:
            pdr_bytes: bytes = s3os.pull_protocoldagresult(
                pdr_sk, transformation_sk, ok=True
            )
        except Exception:
            # if we fail to get the object with the above, fall back to
            # location-based retrieval
            pdr_bytes: bytes = s3os.pull_protocoldagresult(
                location=protocoldagresultref.location,
                ok=True,
            )
        pdr = pdr_bytes.decode("latin-1")
    else:
        pdr = None

    return (gufe_to_json(transformation), pdr)


# TODO: support compression performed client-side
@router.post("/tasks/{task_scoped_key}/results", response_model=ScopedKey)
async def set_task_result(
    task_scoped_key,
    *,
    request: Request,
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    body = await request.body()
    body_ = json.loads(body.decode("utf-8"), cls=JSON_HANDLER.decoder)

    protocoldagresult_ = body_["protocoldagresult"]
    compute_service_id = body_["compute_service_id"]

    # the compute client serializes a missing id as the string "None"; treat
    # that (and a genuine null) as "no compute service", so provenance
    # finalization is simply skipped rather than erroring on an invalid id
    compute_service_id_ = (
        ComputeServiceID(compute_service_id)
        if compute_service_id and compute_service_id != "None"
        else None
    )

    task_sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(task_sk.scope, token)

    pdr: ProtocolDAGResult = decompress_gufe_zstd(protocoldagresult_)

    tf_sk, _ = n4js.get_task_transformation(
        task=task_scoped_key,
        return_gufe=False,
    )

    # push the ProtocolDAGResult to the object store
    protocoldagresultref: ProtocolDAGResultRef = s3os.push_protocoldagresult(
        protocoldagresult=protocoldagresult_,
        protocoldagresult_ok=pdr.ok(),
        protocoldagresult_gufekey=pdr.key,
        transformation=tf_sk,
        creator=compute_service_id,
    )

    # push the reference to the state store; this also finalizes the open
    # TaskProvenance attempt for this (task, compute_service_id) pair with the
    # appropriate outcome and links it to the new ProtocolDAGResultRef
    result_sk: ScopedKey = n4js.set_task_result(
        task=task_sk,
        protocoldagresultref=protocoldagresultref,
        compute_service_id=compute_service_id_,
    )

    # derive one ProtocolUnitResultRef per unit result, and extract any embedded
    # stdout/stderr from the (already-deserialized) ProtocolDAGResult into
    # per-unit, retrieval-optimized artifacts --- flipping has_stdout/has_stderr.
    # Streams need no new compute-facing routes: they ride inside the PDR blob.
    refs_map = n4js.add_protocol_unit_result_refs(protocoldagresultref, result_sk, pdr)
    if protocoldagresultref.location:
        for unit_result in pdr.protocol_unit_results:
            purr_sk = refs_map.get(unit_result.key)
            if purr_sk is None:
                continue
            unit_location = protocol_unit_result_location(
                protocoldagresultref.location, unit_result.key
            )
            if unit_result.stdout:
                s3os.push_protocol_unit_result_streams(
                    unit_location, "stdout", unit_result.stdout
                )
                n4js.set_protocol_unit_result_ref_artifacts(purr_sk, has_stdout=True)
            if unit_result.stderr:
                s3os.push_protocol_unit_result_streams(
                    unit_location, "stderr", unit_result.stderr
                )
                n4js.set_protocol_unit_result_ref_artifacts(purr_sk, has_stderr=True)

    # if success, set task complete, remove from all hubs
    # otherwise, set as errored, leave in hubs
    if protocoldagresultref.ok:
        n4js.set_task_complete(tasks=[task_sk])
    else:
        n4js.add_protocol_dag_result_ref_tracebacks(
            pdr.protocol_unit_failures, result_sk
        )
        # provenance already finalized by set_task_result above, so no
        # compute_service_id passed here (avoids a redundant finalization)
        n4js.set_task_error(tasks=[task_sk])

        # report that the compute service experienced a failure
        now = datetime.datetime.now(tz=datetime.UTC)
        n4js.log_failure_compute_service(compute_service_id, now)
        n4js.resolve_task_restarts(task_scoped_keys=[task_sk])

    return result_sk


@router.post("/tasks/{task_scoped_key}/error", response_model=str)
async def set_task_error(
    task_scoped_key,
    *,
    reason: str = Body(..., embed=True),
    compute_service_id: str = Body(..., embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    """Set a Task to `error` with a `reason`, for `ProtocolDAG` creation failures.

    No `ProtocolDAGResult` exists in this case (creation failed before any unit
    ran), which is exactly why `reason` lives on the `Task`. The open
    `TaskProvenance` attempt for this ``(task, compute_service_id)`` pair is
    finalized with `outcome = error`.

    `ProtocolDAG` creation failures deliberately do not participate in restart
    policies: `resolve_task_restarts` matches against `Tracebacks` nodes only,
    and none exist here, so the Task's restarts are cancelled rather than
    retried --- these tend to be systematic problems with the `Transformation`
    itself, which auto-retrying would only mask.
    """
    task_sk = ScopedKey.from_str(task_scoped_key)
    validate_scopes(task_sk.scope, token)

    n4js.set_task_error(
        tasks=[task_sk],
        reason=reason,
        compute_service_id=ComputeServiceID(compute_service_id),
    )

    # no Tracebacks node exists, so this cancels (does not renew) the Task's
    # restart patterns
    n4js.resolve_task_restarts(task_scoped_keys=[task_sk])

    return str(task_sk)


@router.post("/computeservice/{compute_service_id}/progress")
def update_task_progress(
    compute_service_id,
    *,
    progress: dict[str, dict[str, int]] = Body(...),
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    """Record live progress counts for a service's claimed Tasks.

    The request body is the bare map from `Task` ScopedKey string to
    ``{"units_completed": int, "units_total": int}`` (per the design's transport
    spec) --- one batched request per push event, regardless of claim count.
    NOTE: this is deliberately NOT ``embed``ed; the compute client sends the map
    as the whole body. Like a heartbeat, this route never rejects: updates for
    Tasks the service no longer claims are silently dropped server-side (the
    claim expired mid-flight), and malformed entries are skipped rather than
    500'd.
    """
    progress_ = {
        task_sk: (counts["units_completed"], counts["units_total"])
        for task_sk, counts in progress.items()
        if "units_completed" in counts and "units_total" in counts
    }
    n4js.update_task_progress(ComputeServiceID(compute_service_id), progress_)
    return None


@router.post(
    "/tasks/{task_scoped_key}/results/{protocoldagresultref_scoped_key}/units/{unit_result_key}/artifacts/logs"
)
def set_unit_result_logs(
    task_scoped_key,
    protocoldagresultref_scoped_key,
    unit_result_key,
    *,
    logs: str = Body(..., embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    s3os: S3ObjectStore = Depends(get_s3os_depends),
    token: TokenData = Depends(get_token_data_depends),
):
    """Upload captured log text for a single unit result, flipping `has_logs`.

    Ordered after streams and unit-ref creation (which happen in
    `set_task_result`), so a service dying mid-upload leaves consistent state:
    refs and streams exist, missing logs are simply flagged absent.
    """
    task_sk = ScopedKey.from_str(task_scoped_key)
    pdrr_sk = ScopedKey.from_str(protocoldagresultref_scoped_key)
    # authorize BOTH the Task and the ProtocolDAGResultRef scopes, and require
    # the ref to actually be a result of the Task --- otherwise a caller
    # credentialed for one scope could flip flags / overwrite artifacts on a
    # ref in another scope by pairing it with an in-scope Task path parameter
    validate_scopes(task_sk.scope, token)
    validate_scopes(pdrr_sk.scope, token)

    purr_sk = n4js.get_protocol_unit_result_ref_scoped_key(
        pdrr_sk, GufeKey(unit_result_key), task=task_sk
    )
    if purr_sk is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                f"No ProtocolUnitResultRef for unit result '{unit_result_key}' "
                f"under '{protocoldagresultref_scoped_key}' for task "
                f"'{task_scoped_key}'"
            ),
        )

    protocolunitresultref = n4js.get_gufe(purr_sk)
    s3os.push_protocol_unit_result_logs(protocolunitresultref.location, logs)
    n4js.set_protocol_unit_result_ref_artifacts(purr_sk, has_logs=True)

    return str(purr_sk)


def process_compute_manager_id_string(
    compute_manager_id_string: str,
) -> ComputeManagerID:
    """Try creating a ComputeManagerID from a string representation. Raise HTTPException."""
    try:
        compute_manager_id = ComputeManagerID(compute_manager_id_string)
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    return compute_manager_id


@router.post("/computemanager/{compute_manager_id}/register")
def register_computemanager(
    compute_manager_id,
    *,
    steal: bool = Body(False, embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
):

    compute_manager_id = process_compute_manager_id_string(compute_manager_id)

    now = datetime.datetime.now(tz=datetime.UTC)
    cm_registration = ComputeManagerRegistration(
        name=compute_manager_id.name,
        uuid=compute_manager_id.uuid,
        registered=now,
        last_status_update=now,
        status=ComputeManagerStatus.OK,
        detail="",
        saturation=0,
    )

    try:
        compute_manager_id_ = n4js.register_computemanager(cm_registration, steal=steal)
    except ValueError as e:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    return compute_manager_id_


@router.post("/computemanager/{compute_manager_id}/deregister")
def deregister_computemanager(
    compute_manager_id,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    compute_manager_id = process_compute_manager_id_string(compute_manager_id)
    n4js.deregister_computemanager(compute_manager_id)
    return compute_manager_id


@router.post("/computemanager/{compute_manager_id}/instruction")
def get_instruction_computemanager(
    compute_manager_id,
    *,
    scopes: list[Scope] = Body([], embed=True),
    protocols: list[str] = Body([], embed=True),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    settings: ComputeAPISettings = Depends(get_base_api_settings),
    token: TokenData = Depends(get_token_data_depends),
):
    scopes = scopes or [Scope()]
    scopes_reduced = minimize_scope_space(scopes)
    query_scopes = []
    for scope in scopes_reduced:
        query_scopes.extend(validate_scopes_query(scope, token))

    compute_manager_id = process_compute_manager_id_string(compute_manager_id)
    now = datetime.datetime.now(tz=datetime.UTC)
    instruction, payload = n4js.get_computemanager_instruction(
        compute_manager_id,
        now - timedelta(seconds=settings.ALCHEMISCALE_COMPUTE_API_FORGIVE_TIME_SECONDS),
        settings.ALCHEMISCALE_COMPUTE_API_MAX_FAILURES,
        query_scopes,
        protocols,
    )
    payload["instruction"] = str(instruction)
    return payload


@router.post("/computemanager/{compute_manager_id}/status")
def update_status_computemanager(
    compute_manager_id,
    *,
    status: str = Body(),
    detail: str | None = Body(None),
    saturation: float | None = Body(None),
    n4js: Neo4jStore = Depends(get_n4js_depends),
    settings: ComputeAPISettings = Depends(get_base_api_settings),
):
    expire_seconds = settings.ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS
    expire_seconds_errored = (
        settings.ALCHEMISCALE_COMPUTE_API_MANAGER_EXPIRE_SECONDS_ERROR
    )
    compute_manager_id = process_compute_manager_id_string(compute_manager_id)
    try:
        n4js.update_compute_manager_status(
            compute_manager_id, status, detail, saturation
        )
        now = datetime.datetime.now(tz=datetime.UTC)
        n4js.expire_computemanager_registrations(
            now - timedelta(seconds=expire_seconds),
            now - timedelta(seconds=expire_seconds_errored),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return compute_manager_id


@router.post("/computemanager/{compute_manager_name}/clear_error")
def clear_error_computemanager(
    compute_manager_name: str,
    n4js: Neo4jStore = Depends(get_n4js_depends),
):
    # validate that compute_manager_name is valid
    ComputeManagerID.new_from_name(compute_manager_name)

    with n4js.transaction() as tx:
        compute_manager_id = n4js.get_compute_manager_id(
            name=compute_manager_name, tx=tx
        )
        n4js.clear_errored_computemanager(compute_manager_id, tx=tx)


### add router

app.include_router(router)
