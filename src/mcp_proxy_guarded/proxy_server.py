"""Create an MCP server that proxies requests through an MCP client.

This server is created independent of any transport mechanism.
"""

import logging
from typing import Any

from mcp import server, types
from mcp.client.session import ClientSession

from agent_action_classifier import is_action_harmful

logger = logging.getLogger(__name__)


def format_tool_call_for_classification(
    tool_name: str, 
    arguments: dict[str, Any] | None,
    server_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Format a tool call request into the action dict format expected by the classifier.
    
    Args:
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        server_info: Optional server information (server_label, server_url, etc.)
    
    Returns:
        Action dict in the format expected by ActionClassifier
    """
    action_dict = {
        "label": tool_name,
        "resource": "Tool",
        "action": {
            "type": "mcp",
            "server_label": server_info.get("server_label", "unknown") if server_info else "unknown",
            "server_url": server_info.get("server_url", "") if server_info else "",
            "parameters": arguments or {},
            "require_approval": "never"  # Default, could be configurable
        }
    }
    return action_dict


async def create_guarded_proxy_server(remote_app: ClientSession) -> server.Server[object]:  # noqa: C901, PLR0915
    """Create a server instance from a remote app."""
    logger.debug("Sending initialization request to remote MCP server...")
    response = await remote_app.initialize()
    capabilities = response.capabilities

    logger.debug("Configuring proxied MCP server...")
    app: server.Server[object] = server.Server(name=response.serverInfo.name)

    if capabilities.prompts:
        logger.debug("Capabilities: adding Prompts...")

        async def _list_prompts(_: Any) -> types.ServerResult:  # noqa: ANN401
            result = await remote_app.list_prompts()
            return types.ServerResult(result)

        app.request_handlers[types.ListPromptsRequest] = _list_prompts

        async def _get_prompt(req: types.GetPromptRequest) -> types.ServerResult:
            result = await remote_app.get_prompt(req.params.name, req.params.arguments)
            return types.ServerResult(result)

        app.request_handlers[types.GetPromptRequest] = _get_prompt

    if capabilities.resources:
        logger.debug("Capabilities: adding Resources...")

        async def _list_resources(_: Any) -> types.ServerResult:  # noqa: ANN401
            result = await remote_app.list_resources()
            return types.ServerResult(result)

        app.request_handlers[types.ListResourcesRequest] = _list_resources

        async def _list_resource_templates(_: Any) -> types.ServerResult:  # noqa: ANN401
            result = await remote_app.list_resource_templates()
            return types.ServerResult(result)

        app.request_handlers[types.ListResourceTemplatesRequest] = _list_resource_templates

        async def _read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
            # Filter if URI the request contains harmful parameters
            action_dict = format_tool_call_for_classification(
                tool_name=req.params.uri.path or "resource",
                arguments={
                    key: value for key, value in req.params.uri.query_params()
                },
                server_info={
                    "server_url": str(req.params.uri),
                    "server_label": "resource"
                },
            )

            classification, confidence = is_action_harmful(action_dict)
            if classification:
                logger.warning(
                    f"WARNING: Tool call '{req.params.uri.path}' classified as potentially '{classification}' "
                    f"(confidence: {confidence:.2f}). Proceeding with caution."
                )
                raise ValueError(
                    f"Blocked potentially harmful resource call '{req.params.uri}' "
                )

            result = await remote_app.read_resource(req.params.uri)
            return types.ServerResult(result)

        app.request_handlers[types.ReadResourceRequest] = _read_resource

    if capabilities.logging:
        logger.debug("Capabilities: adding Logging...")

        async def _set_logging_level(req: types.SetLevelRequest) -> types.ServerResult:
            await remote_app.set_logging_level(req.params.level)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SetLevelRequest] = _set_logging_level

    if capabilities.resources:
        logger.debug("Capabilities: adding Resources...")

        async def _subscribe_resource(req: types.SubscribeRequest) -> types.ServerResult:
            await remote_app.subscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SubscribeRequest] = _subscribe_resource

        async def _unsubscribe_resource(req: types.UnsubscribeRequest) -> types.ServerResult:
            await remote_app.unsubscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.UnsubscribeRequest] = _unsubscribe_resource

    if capabilities.tools:
        logger.debug("Capabilities: adding Tools...")

        async def _list_tools(_: Any) -> types.ServerResult:  # noqa: ANN401
            tools = await remote_app.list_tools()
            return types.ServerResult(tools)

        app.request_handlers[types.ListToolsRequest] = _list_tools

        async def _call_tool(req: types.CallToolRequest) -> types.ServerResult:
            try:
                # Classify the action for potential harm
                action_dict = format_tool_call_for_classification(
                    tool_name=req.params.name, 
                    arguments=req.params.arguments,
                    server_info={
                        "server_label": "tool"
                    },
                )

                classification, confidence = is_action_harmful(action_dict)
                if classification:
                    logger.warning(
                        f"WARNING: Tool call '{req.params.name}' classified as potentially '{classification}' "
                        f"(confidence: {confidence:.2f}). Proceeding with caution."
                    )
                    return types.ServerResult(
                        types.CallToolResult(
                            content=[
                                types.TextContent(
                                    type="text",
                                    text=(
                                        f"Tool call '{req.params.name}' blocked due to potential '{classification}'"
                                        f" with confidence {confidence:.2f}. Only safe actions are allowed."
                                    )
                                )
                            ],
                            structuredContent={
                                "blocked": True,
                                "reason": "potentially harmful/unethical action",
                                "classification": classification,
                                "confidence": confidence,
                            },
                            isError=True,
                        ),
                    )

                result = await remote_app.call_tool(
                    req.params.name,
                    (req.params.arguments or {}),
                )
                return types.ServerResult(result)
            except Exception as e:  # noqa: BLE001
                logger.exception("Error while calling remote tool '%s'", req.params.name)
                # Return the exception type and message to make failures easier to diagnose
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"{e.__class__.__name__}: {e}",
                            )
                        ],
                        structuredContent={
                            "exception_type": e.__class__.__name__,
                        },
                        isError=True,
                    ),
                )

        app.request_handlers[types.CallToolRequest] = _call_tool

    async def _send_progress_notification(req: types.ProgressNotification) -> None:
        await remote_app.send_progress_notification(
            req.params.progressToken,
            req.params.progress,
            req.params.total,
        )

    app.notification_handlers[types.ProgressNotification] = _send_progress_notification

    async def _complete(req: types.CompleteRequest) -> types.ServerResult:
        result = await remote_app.complete(
            req.params.ref,
            req.params.argument.model_dump(),
        )
        return types.ServerResult(result)

    app.request_handlers[types.CompleteRequest] = _complete

    return app


async def create_direct_proxy_server(remote_app: ClientSession) -> server.Server[object]:  # noqa: C901, PLR0915
    """Create a direct proxy server without classification.

    This server proxies requests through an MCP client without any filtering.
    """
    logger.debug("Sending initialization request to remote MCP server...")
    response = await remote_app.initialize()
    capabilities = response.capabilities

    logger.debug("Configuring direct proxied MCP server...")
    app: server.Server[object] = server.Server(name=response.serverInfo.name)

    if capabilities.prompts:
        logger.debug("Capabilities: adding Prompts...")

        async def _list_prompts(_: Any) -> types.ServerResult:  # noqa: ANN401
            result = await remote_app.list_prompts()
            return types.ServerResult(result)

        app.request_handlers[types.ListPromptsRequest] = _list_prompts

        async def _get_prompt(req: types.GetPromptRequest) -> types.ServerResult:
            result = await remote_app.get_prompt(req.params.name, req.params.arguments)
            return types.ServerResult(result)

        app.request_handlers[types.GetPromptRequest] = _get_prompt

    if capabilities.resources:
        logger.debug("Capabilities: adding Resources...")

        async def _list_resources(_: Any) -> types.ServerResult:  # noqa: ANN401
            result = await remote_app.list_resources()
            return types.ServerResult(result)

        app.request_handlers[types.ListResourcesRequest] = _list_resources

        async def _list_resource_templates(_: Any) -> types.ServerResult:  # noqa: ANN401
            result = await remote_app.list_resource_templates()
            return types.ServerResult(result)

        app.request_handlers[types.ListResourceTemplatesRequest] = _list_resource_templates

        async def _read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
            result = await remote_app.read_resource(req.params.uri)
            return types.ServerResult(result)

        app.request_handlers[types.ReadResourceRequest] = _read_resource

    if capabilities.logging:
        logger.debug("Capabilities: adding Logging...")

        async def _set_logging_level(req: types.SetLevelRequest) -> types.ServerResult:
            await remote_app.set_logging_level(req.params.level)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SetLevelRequest] = _set_logging_level

    if capabilities.resources:
        logger.debug("Capabilities: adding Resources...")

        async def _subscribe_resource(req: types.SubscribeRequest) -> types.ServerResult:
            await remote_app.subscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SubscribeRequest] = _subscribe_resource

        async def _unsubscribe_resource(req: types.UnsubscribeRequest) -> types.ServerResult:
            await remote_app.unsubscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.UnsubscribeRequest] = _unsubscribe_resource

    if capabilities.tools:
        logger.debug("Capabilities: adding Tools...")

        async def _list_tools(_: Any) -> types.ServerResult:  # noqa: ANN401
            tools = await remote_app.list_tools()
            return types.ServerResult(tools)

        app.request_handlers[types.ListToolsRequest] = _list_tools

        async def _call_tool(req: types.CallToolRequest) -> types.ServerResult:
            try:
                result = await remote_app.call_tool(
                    req.params.name,
                    (req.params.arguments or {}),
                )
                return types.ServerResult(result)
            except Exception as e:  # noqa: BLE001
                logger.exception("Error while calling remote tool '%s' (direct proxy)", req.params.name)
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"{e.__class__.__name__}: {e}",
                            )
                        ],
                        structuredContent={
                            "exception_type": e.__class__.__name__,
                        },
                        isError=True,
                    ),
                )

        app.request_handlers[types.CallToolRequest] = _call_tool

    async def _send_progress_notification(req: types.ProgressNotification) -> None:
        await remote_app.send_progress_notification(
            req.params.progressToken,
            req.params.progress,
            req.params.total,
        )

    app.notification_handlers[types.ProgressNotification] = _send_progress_notification

    async def _complete(req: types.CompleteRequest) -> types.ServerResult:
        result = await remote_app.complete(
            req.params.ref,
            req.params.argument.model_dump(),
        )
        return types.ServerResult(result)

    app.request_handlers[types.CompleteRequest] = _complete

    return app
