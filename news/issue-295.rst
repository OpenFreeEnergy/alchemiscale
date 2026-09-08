**Added:**

* Per-unit log capture, including ``stdout`` and ``stderr``, for ``Task`` executions.
* New client methods for retrieving execution logs and captured output: ``AlchemiscaleClient.get_task_result_recs``, ``AlchemiscaleClient.get_result_unit_recs``, ``AlchemiscaleClient.get_result_unit_logs``, ``AlchemiscaleClient.get_result_unit_stdout``, ``AlchemiscaleClient.get_result_unit_stderr``, ``AlchemiscaleClient.get_result_logs``, ``AlchemiscaleClient.get_task_stdout``, and ``AlchemiscaleClient.get_task_stderr``.
