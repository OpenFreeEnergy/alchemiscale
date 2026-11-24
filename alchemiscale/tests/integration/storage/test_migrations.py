import datetime

from alchemiscale.migrations import v08_to_v09
from alchemiscale.storage.statestore import Neo4jStore


def test_v08_to_v09(n4jstore_settings):
    n4js = Neo4jStore(n4jstore_settings)
    try:
        n4js.reset()

        # a v0.8 database: no constraint on manager names, registrations
        # without a version, an ERROR leftover, and a duplicate name
        now = datetime.datetime.now(tz=datetime.UTC)
        create = """
        CREATE (:ComputeManagerRegistration {
            name: $name, uuid: $uuid, status: $status, detail: $detail,
            saturation: 0.0, registered: datetime($time),
            last_status_update: datetime($time)
        })
        """
        for name, uuid, status, age in [
            ("errored", "a" * 32, "ERROR", 10),
            ("duplicated", "b" * 32, "OK", 60),
            ("duplicated", "c" * 32, "OK", 5),
        ]:
            n4js.execute_query(
                create,
                name=name,
                uuid=uuid,
                status=status,
                detail="broken" if status == "ERROR" else "",
                time=(now - datetime.timedelta(minutes=age)).isoformat(),
            )
        n4js.execute_query(
            """
            CREATE (:ComputeServiceRegistration {
                identifier: $identifier, manager_name: "duplicated",
                registered: datetime(), heartbeat: datetime(), failure_times: []
            })
            """,
            identifier="duplicated.x-" + "d" * 32,
        )

        v08_to_v09.migrate(n4js)

        records = n4js.execute_query("""
            MATCH (cmr:ComputeManagerRegistration)
            OPTIONAL MATCH (cmr)-[:MANAGES]->(csr:ComputeServiceRegistration)
            RETURN cmr.name AS name, cmr.uuid AS uuid, cmr.status AS status,
                   cmr.version AS version, count(csr) AS services
            ORDER BY name
            """).records
        assert [dict(r) for r in records] == [
            # the most recently updated duplicate is kept, with its services
            {
                "name": "duplicated",
                "uuid": "c" * 32,
                "status": "OK",
                "version": 0,
                "services": 1,
            },
            # ERROR registrations are kept, and no longer block registration
            {
                "name": "errored",
                "uuid": "a" * 32,
                "status": "ERROR",
                "version": 0,
                "services": 0,
            },
        ]

        assert n4js.check() is None

        # running it again changes nothing
        v08_to_v09.migrate(n4js)
        assert (
            len(
                n4js.execute_query(
                    "MATCH (cmr:ComputeManagerRegistration) RETURN cmr"
                ).records
            )
            == 2
        )
    finally:
        n4js.reset()
        n4js.initialize()
        n4js.close()
