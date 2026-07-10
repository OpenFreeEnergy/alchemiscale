**Added:**

* Durable per-attempt execution provenance: each ``Task`` execution attempt is now recorded as a ``TaskProvenance`` record, capturing details such as the compute service that claimed it and when.
* Compute services now register with a ``hostname``, recorded alongside their execution provenance.
* ``AlchemiscaleClient.get_task_history`` returns the full per-attempt history of a ``Task``, and ``AlchemiscaleClient.get_tasks_details`` returns detailed per-``Task`` information.

**Changed:**

* ``Task`` records now expose additional status indicators: ``datetime_status_changed`` and ``reason``.
