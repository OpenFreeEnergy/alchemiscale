**Added:**

* Compute services that regularly fail will be denied when they request to claim ``Task``s
  * This behavior is controlled through the ``ALCHEMISCALE_COMPUTE_API_FORGIVE_TIME_SECONDS`` and the ``ALCHEMISCALE_COMPUTE_API_MAX_FAILURES`` ``ComputeAPISettings`` attributes. The former determines how long it takes for a failure to be considered forgiven while the latter dictates the number of allowed failures before a denial occurs.
