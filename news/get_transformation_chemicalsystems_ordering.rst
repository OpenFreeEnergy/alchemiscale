**Changed:**

* Updated ``AlchemiscaleClient.get_transformation_chemicalsystems`` method:
  * Now returns the ``ScopedKeys`` of a given ``Transformation`` ordered by the states they represent:
    * The first ``ScopedKey`` corresponds to the ``ChemicalSystem`` for ``stateA``.
    * The second ``ScopedKey`` corresponds to the ``ChemicalSystem`` for ``stateB``.
  * If provided with a ``NonTransformation`` ``ScopedKey``, the method returns a list containing only the single key representing the ``system``.
