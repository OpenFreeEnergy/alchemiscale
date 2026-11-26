import datetime
from time import sleep

import pytest

from alchemiscale.compute import client
from alchemiscale.migrations import v08_to_v09
from alchemiscale.storage.models import (
    ComputeManagerStateUpdate,
    ComputeManagerID,
    ComputeManagerInstruction,
    ComputeManagerStatus,
    ComputeServiceID,
)
from alchemiscale.tests.integration.compute.utils import get_compute_settings_override


class TestComputeManagerClient:

    def test_wrong_credential(
        self,
        scope_test,
        n4js_preloaded,
        manager_client_wrong_credential: client.AlchemiscaleComputeManagerClient,
        uvicorn_server,
    ):
        with pytest.raises(client.AlchemiscaleComputeManagerClientError):
            manager_client_wrong_credential.get_info()

    def test_refresh_credential(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
        uvicorn_server,
    ):
        settings = get_compute_settings_override()
        assert compute_manager_client._jwtoken is None
        compute_manager_client._get_token()

        token = compute_manager_client._jwtoken
        assert token is not None

        # token shouldn't change this fast
        compute_manager_client.get_info()
        assert token == compute_manager_client._jwtoken

        # should change if we wait a bit
        sleep(settings.JWT_EXPIRE_SECONDS + 2)
        compute_manager_client.get_info()
        assert token != compute_manager_client._jwtoken

    def test_api_check(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
        uvicorn_server,
    ):
        compute_manager_client._api_check()

    def test_registration(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id_1 = ComputeManagerID.new_from_name("testmanager")
        compute_manager_id_2 = ComputeManagerID.new_from_name("testmanager")

        returned_id = compute_manager_client.register(compute_manager_id_1)
        assert compute_manager_id_1 == returned_id

        # a second instance joins the registration
        assert compute_manager_client.register(compute_manager_id_2) == (
            compute_manager_id_1
        )

        # an ERROR registration doesn't block registering
        compute_manager_client.update_status(
            compute_manager_id_1,
            ComputeManagerStatus.ERROR,
            detail="manual error state",
        )
        assert compute_manager_client.register(compute_manager_id_2) == (
            compute_manager_id_1
        )

        # steal is deprecated, and ignored
        with pytest.warns(DeprecationWarning, match="steal"):
            returned_id = compute_manager_client.register(
                compute_manager_id_2, steal=True
            )
        assert returned_id == compute_manager_id_1

    def test_deregistration(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        compute_manager_client.register(compute_manager_id)
        returned_id = compute_manager_client.deregister(compute_manager_id)
        assert compute_manager_id == returned_id

    def test_get_instruction(
        self,
        n4js_preloaded,
        compute_client: client.AlchemiscaleComputeClient,
        compute_service_id: ComputeServiceID,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        compute_manager_client.register(compute_manager_id)

        # no compute services being managed
        instruction, payload = compute_manager_client.get_instruction(
            [], [], compute_manager_id
        )
        assert payload.pop("state").registration.name == "testmanager"

        assert instruction == ComputeManagerInstruction.OK
        assert payload == {"compute_service_ids": [], "num_tasks": 3}

        # try again, but with a protocol filter
        instruction, payload = compute_manager_client.get_instruction(
            [], ["FakeProtocol"], compute_manager_id
        )
        payload.pop("state")

        assert instruction == ComputeManagerInstruction.OK
        assert payload == {"compute_service_ids": [], "num_tasks": 0}

        # try again, name all possible protocols
        instruction, payload = compute_manager_client.get_instruction(
            [],
            ["FakeProtocol", "DummyProtocolA", "DummyProtocolB", "DummyProtocolC"],
            compute_manager_id,
        )
        payload.pop("state")

        assert instruction == ComputeManagerInstruction.OK
        assert payload == {"compute_service_ids": [], "num_tasks": 3}

        # mock the creation of a compute service
        _compute_service_id = compute_client.register(
            compute_service_id, compute_manager_id=compute_manager_id
        )

        instruction, payload = compute_manager_client.get_instruction(
            [], [], compute_manager_id
        )
        assert payload.pop("state").registration.name == "testmanager"

        assert instruction == ComputeManagerInstruction.OK, (instruction, payload)
        assert payload == {
            "compute_service_ids": [_compute_service_id],
            "num_tasks": 3,
        }, (instruction, payload)

        # add in many failures to trigger a SKIP instruction on next request

        add_fake_failures_query = """
        UNWIND $failure_times AS failure_time
        MATCH (csr: ComputeServiceRegistration {identifier: $compute_service_id})
        SET csr.failure_times = datetime(failure_time) + csr.failure_times
        """

        failure_times = [datetime.datetime.now(tz=datetime.UTC)] * 4
        n4js_preloaded.execute_query(
            add_fake_failures_query,
            compute_service_id=compute_service_id,
            failure_times=failure_times,
        )

        instruction, payload = compute_manager_client.get_instruction(
            [], [], compute_manager_id
        )
        assert payload.pop("state").registration.name == "testmanager"

        assert instruction == ComputeManagerInstruction.SKIP, (instruction, payload)
        assert payload == {"compute_service_ids": [_compute_service_id]}

        # change the UUID associated with the stored manager to simulate usurping

        usurp_query = """
        MATCH (cmr: ComputeManagerRegistration {name: $name})
        SET cmr.uuid = $new_uuid
        """
        from uuid import uuid4

        new_uuid = uuid4()
        params = {
            "name": compute_manager_id.name,
            "new_uuid": str(new_uuid),
        }
        n4js_preloaded.execute_query(usurp_query, params)

        # registrations are found by name, so the manager carries on with the
        # registration that replaced its own
        _, payload = compute_manager_client.get_instruction([], [], compute_manager_id)
        assert payload["state"].registration.uuid == str(new_uuid)

        # until there is none under its name
        n4js_preloaded.execute_query(
            "MATCH (cmr:ComputeManagerRegistration) DETACH DELETE cmr"
        )
        with pytest.raises(client.ComputeManagerNotRegisteredError):
            compute_manager_client.get_instruction([], [], compute_manager_id)

    def test_update_status(
        self,
        n4js_preloaded,
        compute_manager_client: client.AlchemiscaleComputeManagerClient,
    ):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        compute_manager_client.register(compute_manager_id)

        query = """
        MATCH (cmr: ComputeManagerRegistration {name: $name})
        RETURN cmr.status AS status, cmr.detail as detail
        """

        params = {"name": compute_manager_id.name}

        # status: OK
        returned_id = compute_manager_client.update_status(
            compute_manager_id,
            ComputeManagerStatus.OK,
            saturation=0,
        )

        assert returned_id == compute_manager_id

        results = n4js_preloaded.execute_query(query, params)
        record = results.records[0]
        assert record["status"] == "OK" and record["detail"] is None

        # status: ERROR
        exception = RuntimeError("UnexpectedError")

        returned_id = compute_manager_client.update_status(
            compute_manager_id,
            ComputeManagerStatus.ERROR,
            detail=repr(exception),
        )

        assert returned_id == compute_manager_id

        results = n4js_preloaded.execute_query(query, params)
        record = results.records[0]
        assert record["status"] == "ERROR" and record["detail"] == repr(exception)


class TestComputeManagerRegistrationStates:
    """Every manager endpoint, called with no registration, a live OK,
    DEGRADED or ERROR registration, a stale registration, and an ERROR
    registration left over from 0.8, under the manager's name."""

    @pytest.fixture
    def manager_id(self, n4js_preloaded, compute_manager_client, request):
        compute_manager_id = ComputeManagerID.new_from_name("testmanager")
        state = request.param
        if state == "migrated v08 ERROR":
            # as 0.8 wrote it, then migrated
            now = datetime.datetime.now(tz=datetime.UTC)
            n4js_preloaded.execute_query(
                """
                CREATE (:ComputeManagerRegistration {
                    name: $name, uuid: $uuid, status: "ERROR", detail: "broken",
                    registered: datetime($now), last_status_update: datetime($now)
                })
                """,
                now=now.isoformat(),
                **compute_manager_id.to_dict(),
            )
            v08_to_v09.migrate(n4js_preloaded)
        elif state != "none":
            compute_manager_client.register(compute_manager_id)
        match state:
            case "ERROR":
                compute_manager_client.update_status(
                    compute_manager_id, ComputeManagerStatus.ERROR, detail="broken"
                )
            case "DEGRADED":
                compute_manager_client.update_status(
                    compute_manager_id,
                    ComputeManagerStatus.DEGRADED,
                    detail="paused",
                    saturation=0,
                )
            case "stale":
                n4js_preloaded.execute_query(
                    """
                    MATCH (cmr:ComputeManagerRegistration {name: $name})
                    SET cmr.last_status_update = datetime($then)
                    """,
                    name=compute_manager_id.name,
                    then=(
                        datetime.datetime.now(tz=datetime.UTC)
                        - datetime.timedelta(days=2)
                    ).isoformat(),
                )
        return state, compute_manager_id

    STATES = ["none", "OK", "DEGRADED", "ERROR", "stale", "migrated v08 ERROR"]
    # states in which the endpoints find no registration
    MISSING = ("none", "stale")

    @pytest.mark.parametrize("manager_id", STATES, indirect=True)
    def test_deregister(self, compute_manager_client, manager_id):
        # a no-op whatever the state, as 0.8 managers expect it to succeed
        state, compute_manager_id = manager_id
        assert compute_manager_client.deregister(compute_manager_id) == (
            compute_manager_id
        )

    @pytest.mark.parametrize("manager_id", STATES, indirect=True)
    def test_get_instruction(self, compute_manager_client, manager_id):
        state, compute_manager_id = manager_id
        if state in self.MISSING:
            with pytest.raises(client.ComputeManagerNotRegisteredError):
                compute_manager_client.get_instruction([], [], compute_manager_id)
        else:
            instruction, _ = compute_manager_client.get_instruction(
                [], [], compute_manager_id
            )
            assert instruction == ComputeManagerInstruction.OK

    @pytest.mark.parametrize("manager_id", STATES, indirect=True)
    def test_update_status(self, compute_manager_client, manager_id):
        state, compute_manager_id = manager_id
        if state in self.MISSING:
            with pytest.raises(client.ComputeManagerNotRegisteredError):
                compute_manager_client.update_status(
                    compute_manager_id, ComputeManagerStatus.OK, saturation=0
                )
        else:
            compute_manager_client.update_status(
                compute_manager_id, ComputeManagerStatus.OK, saturation=0
            )

        # an invalid update is a 400 whatever the registration state
        with pytest.raises(client.AlchemiscaleComputeManagerClientError) as excinfo:
            compute_manager_client.update_status(
                compute_manager_id, ComputeManagerStatus.OK
            )
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("manager_id", STATES, indirect=True)
    def test_clear_error(self, compute_manager_client, manager_id):
        state, compute_manager_id = manager_id
        match state:
            case "none":
                with pytest.raises(client.ComputeManagerNotRegisteredError):
                    compute_manager_client.clear_error(compute_manager_id.name)
            case "OK" | "DEGRADED" | "stale":
                with pytest.raises(
                    client.AlchemiscaleComputeManagerClientError, match="has status"
                ) as excinfo:
                    compute_manager_client.clear_error(compute_manager_id.name)
                assert excinfo.value.status_code == 422
            case "ERROR" | "migrated v08 ERROR":
                with pytest.warns(DeprecationWarning, match="clear_error"):
                    compute_manager_client.clear_error(compute_manager_id.name)
                with pytest.raises(client.ComputeManagerNotRegisteredError):
                    compute_manager_client.get_instruction([], [], compute_manager_id)

    @pytest.mark.parametrize("manager_id", STATES, indirect=True)
    def test_register_steal(self, compute_manager_client, manager_id):
        # steal is ignored: registering joins a live registration
        state, compute_manager_id = manager_id
        new_id = ComputeManagerID.new_from_name(compute_manager_id.name)
        expected = new_id if state in self.MISSING else compute_manager_id
        with pytest.warns(DeprecationWarning):
            assert compute_manager_client.register(new_id, steal=True) == expected

    @pytest.mark.parametrize("manager_id", STATES, indirect=True)
    def test_register(self, compute_manager_client, manager_id):
        # registering always succeeds, joining a live registration
        state, compute_manager_id = manager_id
        new_id = ComputeManagerID.new_from_name(compute_manager_id.name)
        # stale registrations are replaced rather than joined
        expected = new_id if state in self.MISSING else compute_manager_id
        assert compute_manager_client.register(new_id) == expected

    @pytest.mark.parametrize("manager_id", ["OK", "DEGRADED"], indirect=True)
    def test_any_uuid(self, compute_manager_client, manager_id):
        # registrations are found by name, so an instance can keep using its
        # own ID after joining, as 0.8 managers do
        state, compute_manager_id = manager_id
        other_id = ComputeManagerID.new_from_name(compute_manager_id.name)
        assert compute_manager_client.register(other_id) == compute_manager_id
        instruction, data = compute_manager_client.get_instruction([], [], other_id)
        assert data["state"].registration.uuid == compute_manager_id.uuid
        compute_manager_client.update_status(
            other_id, ComputeManagerStatus.OK, saturation=0
        )
        assert compute_manager_client.deregister(other_id) == other_id

    @pytest.mark.parametrize("manager_id", STATES, indirect=True)
    def test_update_state(self, compute_manager_client, manager_id):
        state, compute_manager_id = manager_id
        if state in self.MISSING:
            with pytest.raises(client.ComputeManagerNotRegisteredError):
                compute_manager_client.update_state(
                    compute_manager_id, ComputeManagerStatus.OK, saturation=0
                )
            return

        _, before = compute_manager_client.get_instruction([], [], compute_manager_id)
        version = before["state"].registration.version

        applied, after = compute_manager_client.update_state(
            compute_manager_id,
            ComputeManagerStatus.DEGRADED,
            detail="paused",
            saturation=0,
            state_update=ComputeManagerStateUpdate(
                transition={"pause_cause": "permanent"},
                expected_version=version,
            ),
        )
        assert applied
        assert after.registration.status == ComputeManagerStatus.DEGRADED
        assert after.registration.version == version + 1

        # a transition decided from a stale version is rejected
        applied, after = compute_manager_client.update_state(
            compute_manager_id,
            ComputeManagerStatus.OK,
            saturation=0,
            state_update=ComputeManagerStateUpdate(expected_version=version),
        )
        assert not applied
        assert after.registration.status == ComputeManagerStatus.DEGRADED


class TestComputeManagerStatus:
    def test_get_status(self, n4js_preloaded, compute_manager_client):
        # nothing known about the name
        status = compute_manager_client.get_status("testmanager")
        assert status == {
            "name": "testmanager",
            "state": None,
            "expired": False,
            "failure_count": None,
            "errors": [],
        }

        compute_manager_id = compute_manager_client.register(
            ComputeManagerID.new_from_name("testmanager")
        )
        compute_manager_client.update_status(
            compute_manager_id, ComputeManagerStatus.ERROR, detail="broken"
        )

        status = compute_manager_client.get_status("testmanager")
        assert status["state"].registration.status == ComputeManagerStatus.ERROR
        assert status["state"].registration.detail == "broken"
        assert [e.detail for e in status["errors"]] == ["broken"]
        assert status["failure_count"] == 0
        assert status["state"].registration.identities == [
            compute_manager_client.identifier
        ]

        # an invalid name
        with pytest.raises(client.AlchemiscaleComputeManagerClientError) as excinfo:
            compute_manager_client.get_status("1nvalid-name")
        assert excinfo.value.status_code == 422

    def test_request_probe(self, n4js_preloaded, compute_manager_client):
        with pytest.raises(client.ComputeManagerNotRegisteredError):
            compute_manager_client.request_probe("testmanager")

        compute_manager_id = compute_manager_client.register(
            ComputeManagerID.new_from_name("testmanager")
        )

        # only a paused manager can be asked to probe
        with pytest.raises(client.AlchemiscaleComputeManagerClientError) as excinfo:
            compute_manager_client.request_probe("testmanager")
        assert excinfo.value.status_code == 422

        later = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=1)
        compute_manager_client.update_state(
            compute_manager_id,
            ComputeManagerStatus.DEGRADED,
            detail="paused",
            saturation=0,
            state_update=ComputeManagerStateUpdate(
                transition={
                    "pause_cause": "workload",
                    "backoff_until": later,
                    "probe_job": "testmanager." + "0" * 32,
                }
            ),
        )

        assert compute_manager_client.request_probe("testmanager") == (
            compute_manager_id
        )
        registration = compute_manager_client.get_status("testmanager")[
            "state"
        ].registration
        assert registration.backoff_until < later
        assert registration.probe_job is None
