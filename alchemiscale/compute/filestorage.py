import pathlib
import shutil
import os
from typing import Union, Tuple, ContextManager

from gufe.storage.externalresource.base import ExternalStorage
from gufe.storage.storagemanager import StorageManager, SingleProcDAGContextManager

from gufe.storage.errors import (
    MissingExternalResourceError, ChangedExternalResourceError
)

from ..models import ScopedKey
from .client import AlchemiscaleComputeClient


class ResultFileDAGContextManager(SingleProcDAGContextManager):
    ...


class ResultFileStorageManager(StorageManager):
    ...


class ResultFileStorage(ExternalStorage):

    # need some way of making sure files land in the right place in object store
    # so somehow we need to communicate this in every call to API service, so
    # it can translate what is being requested into the true location in the
    # object store

    # task_sk may be the right thing here, but depends on if paths get shipped
    # *before* or *after* ProtocolDAGResult returned by executor and uploaded
    # it's better for us if paths get shipped *after*, since then we'll have
    # the reference in the state store to use for routing into object store
    def __init__(self, client: AlchemiscaleComputeClient, task_sk: ScopedKey):
        self.client = client

    def _iter_contents(self, prefix=""):
        raise NotImplementedError()

    def _store_bytes(self, location, byte_data):
        """
        For implementers: This should be blocking, even if the storage
        backend allows asynchronous storage.
        """
        raise NotImplementedError()

    def _store_path(self, location, path):
        """
        For implementers: This should be blocking, even if the storage
        backend allows asynchronous storage.
        """
        raise NotImplementedError()

    def _exists(self, location):
        return self.client.check_exists_resultfile(location)

    def _delete(self, location):
        raise NotImplementedError()

    def _get_filename(self, location):
        raise NotImplementedError()

    def _load_stream(self, location):
        raise NotImplementedError()
