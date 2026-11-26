"""ACP 客户端与可续传的 AG-UI 事件输出，不包含 Channel Server 协议。"""

import asyncio
import re
import time
from contextlib import AsyncExitStack, suppress
from pathlib import Path

from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process, text_block
from acp.schema import ClientCapabilities, RequestPermissionResponse
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .config_sync import Snapshot, private_directory
from .events import EventMapper
from .journal import Journal
from .models import Settings

SESSION_VOLUME = Path("/app/.storage/session")


class Adapter:
    def __init__(self, settings: Settings, snapshot: Snapshot):
        self.settings = settings
        self.snapshot = snapshot
        self.journal = Journal(settings.root / "journal", settings.identity)
        self._stack = AsyncExitStack()
        self._connection = None
        self._sessions: dict[str, str] = {}
        self._active: dict[str, tuple[EventMapper, asyncio.Event]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._unrecoverable: set[str] = set()

    def session_directory(self, thread_id: str) -> Path:
        if not self.settings.mounted_storage:
            return self.snapshot.workspace
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", thread_id):
            raise ValueError("invalid session code")
        return private_directory(SESSION_VOLUME / thread_id)

    async def _connect(self) -> None:
        if self._connection is not None:
            return
        command = self.settings.command or [self.settings.engine, "acp"]
        home = private_directory(self.settings.root / "engine-home")
        environment = {
            "HOME": str(home),
            **self.settings.engine_env,
            **self.snapshot.environment,
            "OPENCLAW_HIDE_BANNER": "1",
            "OPENCLAW_SUPPRESS_NOTES": "1",
        }
        async with AsyncExitStack() as startup:
            connection, _ = await startup.enter_async_context(
                spawn_agent_process(
                    self,
                    *command,
                    env=environment,
                    cwd=Path("/app/workspace") if self.settings.mounted_storage else self.snapshot.workspace,
                    transport_kwargs={"stderr": asyncio.subprocess.DEVNULL},
                )
            )
            await connection.initialize(protocol_version=PROTOCOL_VERSION, client_capabilities=ClientCapabilities())
            self._stack.push_async_callback(startup.pop_all().aclose)
        self._connection = connection

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        # 第一版不提供自动授权；后续由 Channel 接入显式用户确认。
        return RequestPermissionResponse(outcome={"outcome": "cancelled"})

    async def session_update(self, session_id, update, **kwargs) -> None:
        active = self._active.get(session_id)
        if active:
            mapper, mapping_failed = active
            try:
                mapper.update(update)
            except Exception:
                mapping_failed.set()

    async def ext_method(self, method, params):
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method, params) -> None:
        pass

    def on_connect(self, conn) -> None:
        pass

    async def submit(self, run_id: str, thread_id: str, text: str, *, queued_ns: int | None = None) -> None:
        if self._closed:
            raise RuntimeError("adapter closed")
        if not text or len(text.encode()) > 1024 * 1024:
            raise ValueError("prompt must be nonempty and at most 1 MiB")
        request = {
            "text": text,
            "revision": self.snapshot.revision,
            "command": self.settings.command,
            "engine_env": self.settings.engine_env,
        }
        if len(self._tasks) >= self.settings.max_pending_runs and not self.journal.contains(run_id):
            raise RuntimeError("adapter request queue is full")
        if self.journal.claim(run_id, thread_id, request):
            now = time.time_ns()
            start = queued_ns if type(queued_ns) is int and 0 < queued_ns <= now else now
            self._tasks[run_id] = asyncio.create_task(self._execute(run_id, thread_id, text, start))
            self._tasks[run_id].add_done_callback(lambda task: self._tasks.pop(run_id, None))

    async def events(self, run_id: str, after: int = 0):
        """断开消费者不会取消执行；携带原 run_id 和 cursor 再次调用即可续传。"""
        while True:
            batch = self.journal.replay(run_id, after)
            for envelope in batch:
                after = envelope["sequence"]
                yield envelope
            if batch:
                continue
            if self.journal.terminal(run_id):
                return
            await asyncio.sleep(0.05)

    async def _execute(self, run_id: str, thread_id: str, text: str, queued_ns: int) -> None:
        tracer = trace.get_tracer(__name__)
        queue_span = tracer.start_span("turn.queue", start_time=queued_ns, attributes={"aidev.run.id": run_id})
        waiting = True
        try:
            async with self._thread_locks.setdefault(thread_id, asyncio.Lock()):
                queue_span.end()
                waiting = False
                with tracer.start_as_current_span("agent.run", attributes={"aidev.run.id": run_id}):
                    await self._run_locked(run_id, thread_id, text)
        except asyncio.CancelledError:
            if waiting:
                queue_span.end()
            # 取消发生在获取串行锁之前，也需要完整 RUN_STARTED / RUN_ERROR。
            self.journal.abort(run_id, "CANCELLED_BEFORE_DISPATCH", "Request cancelled before dispatch")
            raise

    async def _run_locked(self, run_id: str, thread_id: str, text: str) -> None:
        mapper = EventMapper(self.journal, run_id, thread_id)
        mapping_failed = asyncio.Event()
        session_id = None
        try:
            if thread_id in self._unrecoverable or (
                thread_id not in self._sessions and self.journal.has_thread(thread_id, run_id)
            ):
                trace.get_current_span().set_status(trace.StatusCode.ERROR)
                self.journal.abort(run_id, "SESSION_NOT_RECOVERABLE", "ACP session context cannot be recovered")
                return
            async with asyncio.timeout(self.settings.prompt_timeout):
                async with self._connect_lock:
                    await self._connect()
                if thread_id not in self._sessions:
                    session = await self._connection.new_session(
                        cwd=str(self.session_directory(thread_id)), mcp_servers=[]
                    )
                    self._sessions[thread_id] = session.session_id
                session_id = self._sessions[thread_id]
                self._active[session_id] = (mapper, mapping_failed)
                self.journal.mark_dispatched(run_id)
                response = await self._prompt(session_id, text, mapping_failed)
                mapper.finish(response.stop_reason)
        except asyncio.CancelledError:
            self._unrecoverable.add(thread_id)
            self.journal.abort(run_id, "EXECUTION_UNKNOWN", "ACP execution cancelled; outcome unknown")
            raise
        except Exception:
            self._unrecoverable.add(thread_id)
            trace.get_current_span().set_status(trace.StatusCode.ERROR)
            code = "EVENT_MAPPING_FAILED" if mapping_failed.is_set() else "EXECUTION_UNKNOWN"
            self.journal.abort(run_id, code, "ACP execution interrupted; outcome unknown")
        finally:
            if session_id is not None:
                self._active.pop(session_id, None)
                if thread_id in self._unrecoverable:
                    with suppress(Exception):
                        await asyncio.wait_for(self._connection.cancel(session_id=session_id), timeout=2)

    async def _prompt(self, session_id: str, text: str, mapping_failed: asyncio.Event):
        carrier = {}
        if self.settings.engine == "openclaw":
            TraceContextTextMapPropagator().inject(carrier)
        prompt = asyncio.create_task(
            self._connection.prompt(
                session_id=session_id,
                prompt=[text_block(text)],
                **carrier,
            )
        )
        failed = asyncio.create_task(mapping_failed.wait())
        try:
            await asyncio.wait({prompt, failed}, return_when=asyncio.FIRST_COMPLETED)
            if mapping_failed.is_set():
                raise RuntimeError("ACP event conversion failed")
            return await prompt
        finally:
            for task in (prompt, failed):
                if not task.done():
                    task.cancel()
            await asyncio.gather(prompt, failed, return_exceptions=True)

    async def cancel(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task and not task.done():
            # 只取消目标任务；由持有串行锁的任务向 ACP 发送 cancel。
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.journal.abort(run_id, "CANCELLED_BEFORE_DISPATCH", "Request cancelled before dispatch")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        run_ids = list(self._tasks)
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            for run_id in run_ids:
                self.journal.abort(run_id, "EXECUTION_UNKNOWN", "Adapter stopped")
            await self._stack.aclose()
        finally:
            self.journal.close()
