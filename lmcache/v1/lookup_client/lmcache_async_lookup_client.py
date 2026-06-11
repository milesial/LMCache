# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import threading
import time

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import (
    DEFAULT_CLEAR_CACHE_TIMEOUT_MS,
    LookupClientInterface,
)
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupCleanupMsg,
    LookupClearMsg,
    LookupClearResponseMsg,
    LookupRequestMsg,
    LookupResponseMsg,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)

logger = init_logger(__name__)


# NOTE(Jiayi): Prefetch could load extra redundant cache if multiple
# workers has different hit tokens.
class LMCacheAsyncLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup server on all workers
        - if lookup_server_worker_ids is [0], start lookup server on worker0
        - if lookup_server_worker_ids is [0, 3, 6], start lookup server on
          worker0, worker3 and worker6
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        # lookup_id -> first lookup time
        # this helps us support timeout semantics
        self.first_lookup_time: dict[str, float] = {}
        self.config = config

        self.ctx = get_zmq_context(use_asyncio=False)
        kv_connector_extra_config = metadata.kv_connector_extra_config or {}
        rpc_port = kv_connector_extra_config.get("lmcache_rpc_port", 0)
        engine_id = metadata.engine_id
        assert engine_id is not None, "engine_id is required for RPC communication"
        self.world_size = metadata.world_size
        self.lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        self.push_sockets = []
        if len(self.lookup_server_worker_ids) > 0:
            ranks = self.lookup_server_worker_ids
            self.world_size = len(self.lookup_server_worker_ids)
        else:
            ranks = [i for i in range(self.world_size)]

        for rank in ranks:
            worker_socket_path = get_zmq_rpc_path_lmcache(
                engine_id, "lookup_worker", rpc_port, rank
            )
            logger.info(
                "lmcache lookup client connect to rank %s with worker socket path %s",
                rank,
                worker_socket_path,
            )

            push_socket = get_zmq_socket(
                self.ctx,
                worker_socket_path,
                "ipc",
                zmq.PUSH,  # type: ignore[attr-defined]
                "connect",
            )

            self.push_sockets.append(push_socket)

        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            engine_id, "lookup_scheduler", rpc_port, 0
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
        )
        logger.info(
            "lmcache lookup client connect to scheduler with socket path %s",
            scheduler_socket_path,
        )

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.token_database: TokenDatabase
        if config.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

        # A lock is needed since we need another thread to pull
        # responses from the lookup_and_prefetch server
        # (e.g., worker process).
        self.lock = threading.Lock()

        # map from lookup_id (i.e., req_id) to req's status.
        # None indicates ongoing.
        # int indicates number of hit tokens.
        self.reqs_status: dict[str, Optional[int]] = {}

        # map from lookup_id (i.e., req_id) to number of hit tokens for each worker
        self.res_for_each_worker: dict[str, list[int]] = {}

        self.reset_epoch = 0
        self.lookup_epochs: dict[str, int] = {}

        # Clear-cache response tracking for the single blocking clear_cache call.
        self.clear_id: str | None = None
        self.clear_results: list[bool] = []
        self.clear_event = threading.Event()

        # The two parts are [lookup_id (i.e., req_id), num_hit_tokens]
        self.num_parts = 2

        # Track lookup_ids that have been aborted for cleanup
        self.aborted_lookups: set[str] = set()

        self.running = True

        self.thread = threading.Thread(
            target=self.process_responses_from_workers,
            daemon=True,
            name="async-lookup-client-thread",
        )
        self.thread.start()

        # default backoff time
        self.lookup_backoff_time = 0.01
        self.clear_timeout_ms = DEFAULT_CLEAR_CACHE_TIMEOUT_MS
        if config.extra_config is not None:
            self.lookup_backoff_time = float(
                config.extra_config.get("lookup_backoff_time", self.lookup_backoff_time)
            )
            self.clear_timeout_ms = int(
                config.extra_config.get("clear_cache_timeout_ms", self.clear_timeout_ms)
            )

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        -1 means not found;
        None means ongoing;
        int >= 0 means number of hit tokens
        """
        # Check if any aborted lookups are finished, send cleanup messages
        self._cleanup_finished_aborted_lookups()

        with self.lock:
            if (req_status := self.reqs_status.get(lookup_id, -1)) == -1:
                self.reqs_status[lookup_id] = None
                self.first_lookup_time[lookup_id] = time.time()
            elif req_status is None:
                time.sleep(self.lookup_backoff_time)
                if (
                    time.time() - self.first_lookup_time[lookup_id]
                ) * 1000 > self.config.lookup_timeout_ms:
                    logger.warning(
                        (
                            "Request %s is still waiting for async lookup "
                            "after %d seconds, returning 0 lmcache cached tokens "
                            "so vllm can recompute"
                        ),
                        lookup_id,
                        self.config.lookup_timeout_ms // 1000,
                    )
                    self.cancel_lookup(lookup_id)
                    self.first_lookup_time.pop(lookup_id, None)
                    self.lookup_epochs.pop(lookup_id, None)
                    return 0

            return req_status

    # TODO(Jiayi): Consider batching here
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        hashes: list[int] = []
        offsets = []
        for start, end, hash_val in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(hash_val)  # type: ignore[arg-type]
            offsets.append(end - start)

        lookup_epoch = self.reset_epoch
        self.lookup_epochs[lookup_id] = lookup_epoch

        # Create structured message
        msg = LookupRequestMsg(
            lookup_id=lookup_id,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
            lookup_epoch=lookup_epoch,
        )

        # Serialize message using msgspec
        msg_buf = msgspec.msgpack.encode(msg)

        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)
        time.sleep(self.lookup_backoff_time)
        return None

    def process_responses_from_workers(self):
        while self.running:
            try:
                msg_buf = self.pull_socket.recv(copy=False)
                # Deserialize message using msgspec
                msg = msgspec.msgpack.decode(
                    msg_buf,
                    type=LookupResponseMsg | LookupClearResponseMsg,
                )

                if isinstance(msg, LookupResponseMsg):
                    lookup_id = msg.lookup_id
                    res = msg.num_hit_tokens

                    with self.lock:
                        if self.lookup_epochs.get(lookup_id) != msg.lookup_epoch:
                            continue
                        if lookup_id not in self.res_for_each_worker:
                            self.res_for_each_worker[lookup_id] = [res]
                        else:
                            self.res_for_each_worker[lookup_id].append(res)
                        all_res = self.res_for_each_worker[lookup_id]

                        if len(all_res) == self.world_size:
                            self.res_for_each_worker.pop(lookup_id)

                            # NOTE: it is possible that the number of hit
                            # tokens is different across (TP and PP) ranks, so we
                            # can use the minimum value as the number of
                            # hit tokens.
                            self.reqs_status[lookup_id] = min(all_res)
                            self.lookup_epochs.pop(lookup_id, None)
                elif isinstance(msg, LookupClearResponseMsg):
                    with self.lock:
                        if msg.clear_id == self.clear_id:
                            self.clear_results.append(msg.success)
                            if len(self.clear_results) == self.world_size:
                                self.clear_event.set()

            except Exception as e:
                logger.error("Error processing response from worker: %s", e)

    def clear_lookup_status(self, lookup_id: str) -> None:
        with self.lock:
            self.reqs_status.pop(lookup_id, None)
            self.first_lookup_time.pop(lookup_id, None)
            self.lookup_epochs.pop(lookup_id, None)

    def clear_cache(self) -> bool:
        """Clear worker-side LMCache contents and local lookup state.

        Returns:
            True when all workers report a successful clear, False on timeout
            or if any worker reports a failure.
        """
        clear_id = f"clear-{time.time_ns()}"
        msg = LookupClearMsg(clear_id=clear_id)
        msg_buf = msgspec.msgpack.encode(msg)

        with self.lock:
            aborted_lookups = list(self.aborted_lookups)
            self.first_lookup_time.clear()
            self.reqs_status.clear()
            self.res_for_each_worker.clear()
            self.reset_epoch += 1
            self.lookup_epochs.clear()
            self.aborted_lookups.clear()
            self.clear_id = clear_id
            self.clear_results.clear()
            self.clear_event.clear()

        for lookup_id in aborted_lookups:
            self._send_cleanup_message(lookup_id)

        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)

        if self.clear_event.wait(self.clear_timeout_ms / 1000):
            with self.lock:
                results = self.clear_results.copy()
                self.clear_results.clear()
                self.clear_id = None
            if not all(results):
                logger.warning(
                    "Clear cache failed on at least one worker: %s.",
                    results,
                )
                return False
            return True

        with self.lock:
            self.clear_id = None
            self.clear_results.clear()
            self.clear_event.clear()
        logger.warning(
            "Clear cache timed out after %d ms.",
            self.clear_timeout_ms,
        )
        return False

    def cancel_lookup(self, lookup_id: str) -> None:
        """Mark lookup as aborted. Cleanup will happen after task finishes."""
        self.aborted_lookups.add(lookup_id)

    def _cleanup_finished_aborted_lookups(self) -> None:
        """Check for finished aborted lookups and send cleanup messages to workers."""
        # A lookup whose status is None is still loading.
        # We wait for it to finish before cleanup.
        finished_lookups = [
            lookup_id
            for lookup_id in self.aborted_lookups
            if self.reqs_status.get(lookup_id) is not None
        ]
        if finished_lookups:
            self.aborted_lookups.difference_update(finished_lookups)

        # Tell the server to free the reserved memory buffers for each aborted lookup.
        for lookup_id in finished_lookups:
            self._send_cleanup_message(lookup_id)
            self.clear_lookup_status(lookup_id)

    def _send_cleanup_message(self, lookup_id: str) -> None:
        """Send cleanup message to workers to release memory objects."""
        msg = LookupCleanupMsg(lookup_id=lookup_id)
        msg_buf = msgspec.msgpack.encode(msg)

        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)
        logger.debug("Sent cleanup message for lookup_id=%s", lookup_id)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for s in self.push_sockets:
                s.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning("Failed to join thread during close: %s", e)


class LMCacheAsyncLookupServer:
    """ZMQ-based async lookup server that handles lookup and prefetch
    requests using LMCacheEngine."""

    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        kv_connector_extra_config = metadata.kv_connector_extra_config or {}
        rpc_port = kv_connector_extra_config.get("lmcache_rpc_port", 0)
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )
        worker_socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id, "lookup_worker", rpc_port, metadata.worker_id
        )
        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id, "lookup_scheduler", rpc_port, 0
        )
        self.push_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PUSH,  # type: ignore[attr-defined]
            "connect",
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            worker_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
        )

        self.lmcache_engine = lmcache_engine
        self.lookup_epochs: dict[str, int] = {}
        self.running = True

        logger.info(
            "lmcache lookup server start with"
            " scheduler socket path %s, "
            "worker socket path %s",
            scheduler_socket_path,
            worker_socket_path,
        )
        self.thread = threading.Thread(
            target=self.process_requests_from_scheduler,
            daemon=True,
            name="async-lookup-server-thread",
        )
        self.thread.start()

    def process_requests_from_scheduler(self):
        while self.running:
            try:
                msg_buf = self.pull_socket.recv(copy=False)
                msg = msgspec.msgpack.decode(
                    msg_buf,
                    type=LookupRequestMsg | LookupCleanupMsg | LookupClearMsg,
                )

                if isinstance(msg, LookupRequestMsg):
                    # Handle lookup request
                    self.lookup_epochs[msg.lookup_id] = msg.lookup_epoch
                    self.lmcache_engine.async_lookup_and_prefetch(
                        lookup_id=msg.lookup_id,
                        hashes=msg.hashes,
                        offsets=msg.offsets,
                        pin=True,
                        request_configs=msg.request_configs,
                    )

                elif isinstance(msg, LookupCleanupMsg):
                    # Handle cleanup request - release memory objects for aborted lookup
                    self.lookup_epochs.pop(msg.lookup_id, None)
                    self.lmcache_engine.cleanup_memory_objs(msg.lookup_id)

                elif isinstance(msg, LookupClearMsg):
                    try:
                        self.lookup_epochs.clear()
                        self.lmcache_engine.clear()
                        success = True
                    except Exception:
                        logger.exception("Error clearing LMCache")
                        success = False
                    self.send_clear_response_to_scheduler(msg.clear_id, success)

                else:
                    logger.warning("Unknown message type: %s", type(msg))

            except Exception as e:
                logger.error("Error processing request from scheduler: %s", e)

    def send_response_to_scheduler(self, lookup_id: str, num_hit_tokens: int):
        # Create structured response message
        msg = LookupResponseMsg(
            lookup_id=lookup_id,
            lookup_epoch=self.lookup_epochs.pop(lookup_id, 0),
            num_hit_tokens=num_hit_tokens,
        )

        # Serialize message using msgspec
        msg_buf = msgspec.msgpack.encode(msg)
        self.push_socket.send(msg_buf, copy=False)

    def send_clear_response_to_scheduler(self, clear_id: str, success: bool) -> None:
        """Send a clear-cache response to the scheduler."""
        msg = LookupClearResponseMsg(
            clear_id=clear_id,
            success=success,
        )

        msg_buf = msgspec.msgpack.encode(msg)
        self.push_socket.send(msg_buf, copy=False)

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            self.push_socket.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning("Failed to join thread during close: %s", e)
