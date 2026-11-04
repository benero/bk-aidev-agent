# -*- coding: utf-8 -*-
"""OpenClaw 网关 WebSocket 传输：驱动一次对话并产出结构化事件。

为什么不用 HTTP：两个 HTTP 端点都只给助手文本，拿不到内部工具（MCP、exec）活动。
``/v1/chat/completions`` 只有正文；``/v1/responses`` 的 function_call 输出项仅在
"由调用方执行工具"的模式下产生，而这里工具是 OpenClaw 自己执行的。工具事件由网关
按 ``registerToolEventRecipient(runId, connId)`` 定向投递给**发起该 run 的连接**，
旁听收不到，所以要拿工具事件就得自己用 WS 发起对话。

两个易踩空的点（缺任一都表现为"连上了但没有工具事件"）：

- 握手必须声明 ``tool-events`` 能力，否则网关静默不投工具帧；
- ``client.id`` / ``client.mode`` 受枚举约束，``mode: "operator"`` 会被 schema 拒
  （operator 是 role 不是 mode）。

本模块只做协议搬运，产出中立的 ``OpenClawEvent``，AG-UI 语义留给调用方翻译。
"""

import ipaddress
import json
import re
import time
import uuid
from contextlib import suppress
from logging import getLogger
from typing import Generator, Optional
from urllib.parse import urlparse

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None

logger = getLogger(__name__)

# 取自内核 GATEWAY_CLIENT_IDS / GATEWAY_CLIENT_MODES / GATEWAY_CLIENT_CAPS。
# 用 gateway-client/backend 这组身份：协议文档载明同进程后端客户端在 loopback 上
# 以共享网关令牌鉴权时可免 device 配对，正是本代理与内核同容器的形态。
_CLIENT_ID = "gateway-client"
_CLIENT_MODE = "backend"
_CAP_TOOL_EVENTS = "tool-events"

_HANDSHAKE_TIMEOUT = 20.0
_ABORT_TIMEOUT = 10.0
_HISTORY_LIMIT = 1000
_HISTORY_MAX_CHARS = 500_000
_TERMINAL_POLL_INTERVAL = 1.0
_SUCCESS_STOP_REASONS = {"", "complete", "completed", "end_turn", "stop", "success"}
_CANCEL_STOP_REASONS = {"abort", "aborted", "cancel", "cancelled", "canceled"}


def _is_loopback_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class OpenClawEvent:
    """网关事件的中立表示。

    ``kind`` 取 text / tool / thinking / done / error，其余字段按 kind 取用。
    """

    __slots__ = ("kind", "text", "phase", "tool_call_id", "tool_name", "args", "result", "is_error", "raw")

    def __init__(
        self,
        kind,
        *,
        text="",
        phase="",
        tool_call_id="",
        tool_name="",
        args=None,
        result=None,
        is_error=False,
        raw=None,
    ):
        self.kind = kind
        self.text = text
        self.phase = phase
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.args = args
        self.result = result
        self.is_error = is_error
        self.raw = raw

    def __repr__(self):  # 便于排障时直接打印
        return f"<OpenClawEvent {self.kind} phase={self.phase!r} tool={self.tool_name!r}>"


class OpenClawWSError(RuntimeError):
    pass


class OpenClawHistoryError(OpenClawWSError):
    """历史无法安全对齐；消息均为本地生成，可直接向调用方说明恢复限制。"""


class OpenClawWSSession:
    """一次对话 = 一条连接。用完即弃，避免跨请求共享状态。"""

    def __init__(self, url: str, token: str, timeout: float = 300.0, *, model: str = "openclaw"):
        self.url = url
        self.token = token
        self.timeout = timeout
        self.model = model
        self._ws = None
        self._run_id = ""
        self._chat_request_id = ""
        self._closed = False
        self._cancel_requested = False
        self._terminated = False
        self._run_deadline = None

    # ---------------- 连接与握手 ----------------

    def connect(self) -> None:
        """握手；必要时以本地后端身份免鉴权重连。

        网关对"未绑定设备"的连接会清空声明的 scopes 但**仍然放行**，症状是握手
        ``ok: true``、随后 ``chat.send`` 报 ``missing scope``。同容器 loopback 上的
        ``gateway-client``/``backend`` 客户端可豁免设备配对：共享密钥匹配、或干脆
        不带鉴权。所以先带令牌试，授予的 scope 不足时按后者重连。
        """
        granted = self._handshake(with_auth=True)
        if "operator.write" in granted:
            return
        if not _is_loopback_url(self.url):
            raise OpenClawWSError("仅 loopback OpenClaw 网关允许免鉴权后端重连")
        logger.warning("[OPENCLAW] 令牌未获授权（授予 scopes=%s），改以本地后端身份免鉴权重连", granted or [])
        self.close()
        self._closed = False
        granted = self._handshake(with_auth=False)
        if "operator.write" not in granted:
            raise OpenClawWSError(f"网关未授予 operator.write（授予 {granted or []}）")

    def _handshake(self, *, with_auth: bool) -> list:
        if websocket is None:
            raise OpenClawWSError("缺少 websocket-client 依赖，无法走 WS 传输")

        # 必须抑制 Origin：websocket-client 默认按 URL 自动带上，网关据此判定为
        # 浏览器来源，本地后端免配对豁免随即失效——表现为带令牌时 scopes 被清空、
        # 不带令牌时直接 "device identity required"。
        handshake_timeout = min(float(self.timeout), _HANDSHAKE_TIMEOUT)
        self._ws = websocket.create_connection(self.url, timeout=handshake_timeout, suppress_origin=True)
        # 网关先推 connect.challenge，收到后才发 connect。每次 recv 都使用剩余
        # 握手预算，避免 websocket 的对话超时把握手卡住数分钟。
        deadline = time.monotonic() + handshake_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OpenClawWSError("握手超时")
                self._ws.settimeout(remaining)
                try:
                    msg = self._recv_json()
                except Exception as exc:
                    if time.monotonic() >= deadline:
                        raise OpenClawWSError("握手超时") from exc
                    raise OpenClawWSError(f"握手连接中断: {type(exc).__name__}") from exc
                if msg is None:
                    continue
                if msg.get("event") == "connect.challenge":
                    self._send_connect(with_auth=with_auth)
                elif msg.get("type") == "res":
                    if not msg.get("ok"):
                        err = (msg.get("error") or {}).get("message", "unknown")
                        raise OpenClawWSError(f"握手被拒: {err}")
                    auth = (msg.get("payload") or {}).get("auth") or {}
                    return list(auth.get("scopes") or [])
        finally:
            if self._ws is not None:
                self._ws.settimeout(self.timeout)

    def _send_connect(self, *, with_auth: bool) -> None:
        params = {
            "minProtocol": 3,
            "maxProtocol": 4,
            "client": {"id": _CLIENT_ID, "version": "1.0.0", "platform": "linux", "mode": _CLIENT_MODE},
            "role": "operator",
            "scopes": ["operator.read", "operator.write"],
            # 不声明这个能力，网关不会投工具帧（握手依然成功）
            "caps": [_CAP_TOOL_EVENTS],
            "commands": [],
            "permissions": {},
            "locale": "zh-CN",
            "userAgent": "bkai-openclaw-proxy/1.0.0",
        }
        if with_auth and self.token:
            params["auth"] = {"token": self.token}
        self._send({"type": "req", "id": uuid.uuid4().hex, "method": "connect", "params": params})

    # ---------------- 收发 ----------------

    def _send(self, payload: dict) -> None:
        self._ws.send(json.dumps(payload))

    def _recv_json(self) -> Optional[dict]:
        raw = self._ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except ValueError:
            return None  # ping 等非 JSON 帧不该中断消费

    def session_key(self, session_key: str) -> str:
        """与 OpenClaw http-utils 的 routing model 别名保持一致。

        chat.* schema 不接受 agentId；Agent 通过 agent:<id>:<session> 选择。
        裸键由网关解析配置的默认 Agent，不能在 SDK 中假定默认名为 main。
        """
        model = self.model.strip().lower()
        if model in {"", "openclaw", "openclaw/default"}:
            if session_key.lower().startswith("agent:"):
                raise OpenClawWSError("默认 Agent 模型不能与显式 Agent 会话键混用")
            return session_key
        match = re.fullmatch(r"(?:openclaw[:/]|agent:)([a-z0-9][a-z0-9_-]{0,63})", model)
        if not match:
            raise OpenClawWSError("WS model 必须使用 openclaw 或 openclaw/<agentId> 别名")
        agent_id = match.group(1)
        if session_key.lower().startswith("agent:"):
            parts = session_key.split(":", 2)
            if len(parts) != 3 or parts[1].lower() != agent_id or not parts[2]:
                raise OpenClawWSError("model 与会话键的 Agent 不一致")
            return f"agent:{agent_id}:{parts[2]}"
        return f"agent:{agent_id}:{session_key}"

    def send_chat(self, session_key: str, message: str) -> str:
        """发起一次对话，返回本次 run 的 id（用于过滤事件）。"""
        self._run_id = str(uuid.uuid4())
        self._cancel_requested = False
        self._terminated = False
        params = {
            "sessionKey": self.session_key(session_key),
            "message": message,
            "idempotencyKey": self._run_id,
        }
        self._run_deadline = time.monotonic() + self.timeout
        self._chat_request_id = uuid.uuid4().hex
        self._send({"type": "req", "id": self._chat_request_id, "method": "chat.send", "params": params})
        return self._run_id

    def _read_history(self, session_key: str, *, raw: bool, deadline: float) -> list[dict]:
        request_id = uuid.uuid4().hex
        params = {"key" if raw else "sessionKey": self.session_key(session_key), "limit": _HISTORY_LIMIT}
        if not raw:
            params["maxChars"] = _HISTORY_MAX_CHARS
        self._send(
            {"type": "req", "id": request_id, "method": "sessions.get" if raw else "chat.history", "params": params}
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenClawHistoryError("核对 OpenClaw 会话历史超时")
            self._ws.settimeout(remaining)
            msg = self._recv_json()
            if not msg or msg.get("type") != "res" or msg.get("id") != request_id:
                continue
            if not msg.get("ok"):
                raise OpenClawHistoryError("无法核对 OpenClaw 会话历史，拒绝丢失历史继续执行")
            messages = (msg.get("payload") or {}).get("messages")
            if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
                raise OpenClawHistoryError("OpenClaw 会话历史响应格式不正确")
            return messages

    @staticmethod
    def _history_matches(messages: list[dict], expected: list[dict], *, raw: bool) -> bool:
        actual = []
        for message in messages:
            metadata = message.get("__openclaw") or {}
            if metadata.get("truncated"):
                return False
            if metadata.get("kind") == "compaction":
                continue
            role = message.get("role")
            if role not in {"user", "assistant", "system"}:
                continue
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                )
            if not isinstance(content, str):
                return False
            if content or role == "user":
                actual.append({"role": role, "content": content})

        target = expected
        if raw and len(messages) == _HISTORY_LIMIT:
            # seq 是 readSessionMessages 从1开始逐条赋予的源序号，含tool/compaction。
            # 只有完整、连续、达到请求上限的原始窗口才能证明前缀由RPC裁去。
            sequence = [(item.get("__openclaw") or {}).get("seq") for item in messages]
            if all(isinstance(seq, int) and not isinstance(seq, bool) for seq in sequence):
                first = sequence[0]
                if first > 1 and sequence == list(range(first, first + _HISTORY_LIMIT)):
                    if not actual or len(actual) > len(expected):
                        return False
                    target = expected[-len(actual) :]
        if len(actual) != len(target):
            return False
        for observed, wanted in zip(actual, target):
            if observed == wanted:
                continue
            if not raw or observed["role"] != "user" or wanted["role"] != "user":
                return False
            # chat.send唯一自动注入的前缀（agent-timestamp.ts）；不复制UI清洗器，
            # 也不对任意文本作endswith匹配。原文相等优先，用户自带时间戳不会被删。
            unstamped = re.sub(
                r"^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2}(?:[^\]\r\n]*)\] ",
                "",
                observed["content"],
                count=1,
            )
            if unstamped != wanted["content"]:
                return False
        return True

    def verify_history(self, session_key: str, expected: list[dict]) -> None:
        """核验最多1000条原始记录窗口；展示裁剪不被误认成内核丢历史。"""
        deadline = time.monotonic() + min(float(self.timeout), _HANDSHAKE_TIMEOUT)
        try:
            messages = self._read_history(session_key, raw=False, deadline=deadline)
            if self._history_matches(messages, expected, raw=False):
                return
            # chat.history有文本、工具和总字节展示裁剪。用同一有限窗口的只读原始
            # sessions.get回退，不信占位符，也不盲目信任相等后缀。
            messages = self._read_history(session_key, raw=True, deadline=deadline)
            if not self._history_matches(messages, expected, raw=True):
                raise OpenClawHistoryError("OpenClaw 会话历史缺失或不一致，当前 WS 不支持安全恢复；请使用 HTTP 传输")
        finally:
            if self._ws is not None:
                self._ws.settimeout(self.timeout)

    # ---------------- 事件消费 ----------------

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def terminated(self) -> bool:
        return self._terminated

    def events(self) -> Generator[OpenClawEvent, None, None]:
        """消费run事件；attempt终态还需agent.wait零等待快照确认整个chat已结束。"""
        deadline = self._run_deadline or (time.monotonic() + self.timeout)
        candidate = None
        wait_request_id = ""
        next_poll = 0.0
        while True:
            now = time.monotonic()
            if self._run_deadline is not None:
                deadline = min(deadline, self._run_deadline)
            if now >= deadline:
                yield OpenClawEvent("error", text="等待内核运行终态超时")
                return
            if candidate is not None and now >= next_poll:
                wait_request_id = uuid.uuid4().hex
                self._send(
                    {
                        "type": "req",
                        "id": wait_request_id,
                        "method": "agent.wait",
                        "params": {"runId": self._run_id, "timeoutMs": 0},
                    }
                )
                next_poll = now + _TERMINAL_POLL_INTERVAL
            if self._ws is not None:
                # 短轮询让另一个线程的abort截止时间在阻塞recv后也能及时生效；
                # 不依赖跨线程settimeout能否中断已经进入的socket读取。
                read_timeout = min(deadline - now, _TERMINAL_POLL_INTERVAL)
                if candidate is not None:
                    read_timeout = min(read_timeout, max(0.01, next_poll - now))
                self._ws.settimeout(read_timeout)
            try:
                msg = self._recv_json()
            except Exception as exc:
                if self._closed:
                    return
                if isinstance(exc, TimeoutError) or (
                    websocket is not None and isinstance(exc, websocket.WebSocketTimeoutException)
                ):
                    # 一个wait快照响应丢失后允许下一次查询；全程受绝对期限约束。
                    wait_request_id = ""
                    continue
                yield OpenClawEvent("error", text=f"连接中断: {exc}")
                return
            if msg is None:
                continue

            if msg.get("type") == "res":
                if wait_request_id and msg.get("id") == wait_request_id:
                    wait_request_id = ""
                    if not msg.get("ok"):
                        yield OpenClawEvent("error", text="无法确认内核运行终态")
                        return
                    payload = msg.get("payload") or {}
                    if payload.get("runId") != self._run_id:
                        yield OpenClawEvent("error", text="内核运行终态标识不匹配")
                        return
                    status = payload.get("status")
                    if status in {"ok", "error"}:
                        self._terminated = True
                        if status == "error":
                            yield OpenClawEvent("error", text=str(payload.get("error") or "OpenClaw运行失败"))
                        else:
                            yield (
                                candidate
                                if candidate is not None and candidate.kind == "cancelled"
                                else OpenClawEvent("done")
                            )
                        return
                    if status != "timeout":
                        yield OpenClawEvent("error", text="未知的内核运行终态")
                        return
                    continue
                if msg.get("id") == self._chat_request_id and not msg.get("ok"):
                    err = (msg.get("error") or {}).get("message", "unknown")
                    yield OpenClawEvent("error", text=err, raw=msg)
                    return
                continue

            if msg.get("event") == "chat":
                payload = msg.get("payload") or {}
                if payload.get("runId") == self._run_id:
                    state = payload.get("state")
                    if state in {"final", "error", "aborted"}:
                        candidate = OpenClawEvent(
                            {"final": "done", "error": "error", "aborted": "cancelled"}[state],
                            text=str(payload.get("errorMessage") or ""),
                        )
                continue
            if msg.get("event") != "agent":
                continue
            payload = msg.get("payload") or {}
            if not self._run_id or payload.get("runId") != self._run_id:
                continue
            data = payload.get("data") or {}
            if payload.get("stream") == "lifecycle" and data.get("phase") == "start":
                # 模型fallback/重试开始；旧attempt error不可作为最终结果。
                candidate = None
                wait_request_id = ""
            event = self._translate(payload)
            if event is None:
                continue
            if event.kind in {"done", "cancelled", "error"}:
                candidate = event
                continue
            yield event

    def _translate(self, payload: dict) -> Optional[OpenClawEvent]:
        """把网关 agent 事件翻成中立事件。

        实测的帧词汇（``stream`` / ``data.phase``）：

        - ``lifecycle`` start|end|error 是attempt边界；整个chat终态由agent.wait核验。
          别拿其它流的 ``phase == "end"`` 当运行结束，那只是单个条目收尾。
        - ``assistant`` —— 正文增量在 ``data.delta``，**不带 phase**。
        - ``tool`` start|update|result —— ``result`` 阶段才带 ``data.result``
          与 ``isError``；``update`` 是执行中的局部输出，忽略以免重复计数。
        - ``item`` / ``command_output`` —— 面向 UI 的条目与命令输出，与 tool 流
          重复，这里不用。
        """
        stream = payload.get("stream")
        data = payload.get("data") or {}
        phase = data.get("phase") or ""

        if stream == "lifecycle":
            if phase == "error":
                return OpenClawEvent(
                    "error",
                    text=str(data.get("error") or data.get("message") or "OpenClaw运行失败"),
                    phase=phase,
                    raw=payload,
                )
            if phase == "end":
                stop_reason = str(data.get("stopReason") or "").strip().lower()
                error = data.get("error") or data.get("message")
                if stop_reason in _CANCEL_STOP_REASONS and not error:
                    return OpenClawEvent("cancelled", phase=phase, raw=payload)
                if error or stop_reason not in _SUCCESS_STOP_REASONS:
                    detail = str(error or stop_reason or "unknown")
                    return OpenClawEvent("error", text=detail, phase=phase, raw=payload)
                return OpenClawEvent("done", phase=phase, raw=payload)
            return None

        if stream == "tool":
            if phase == "update":
                return None
            call_id = str(data.get("toolCallId") or data.get("callId") or "")
            return OpenClawEvent(
                "tool",
                phase=phase,
                tool_call_id=call_id,
                tool_name=data.get("name") or data.get("toolName") or "",
                args=data.get("args"),
                result=data.get("result"),
                is_error=bool(data.get("isError")),
                raw=payload,
            )

        if stream in ("assistant", "thinking"):
            text = data.get("delta")
            if isinstance(text, str) and text:
                return OpenClawEvent("thinking" if stream == "thinking" else "text", text=text, raw=payload)
        return None

    # ---------------- 收尾 ----------------

    def abort(self, session_key: str) -> bool:
        if self._closed or not self._run_id:
            return False
        try:
            self._send(
                {
                    "type": "req",
                    "id": uuid.uuid4().hex,
                    "method": "chat.abort",
                    "params": {"sessionKey": self.session_key(session_key), "runId": self._run_id},
                }
            )
            self._cancel_requested = True
            abort_deadline = time.monotonic() + _ABORT_TIMEOUT
            self._run_deadline = min(self._run_deadline or abort_deadline, abort_deadline)
            if self._ws is not None:
                self._ws.settimeout(min(float(self.timeout), _ABORT_TIMEOUT))
            return True
        except Exception as exc:
            logger.warning("[OPENCLAW] abort 发送失败: %s", exc)
            return False

    def cancel(self, session_key: str) -> bool:
        """请求内核终止运行；发送失败时关闭传输，由调用方隔离身份租约。"""
        sent = self.abort(session_key)
        if not sent:
            self.close()
        return sent

    def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            with suppress(Exception):
                self._ws.close()
            self._ws = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
