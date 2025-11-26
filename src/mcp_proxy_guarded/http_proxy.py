"""HTTP-to-HTTP proxy with harmful action classification."""

import logging
from functools import partial
from typing import Any

import httpx
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Mount, Route
from starlette.types import Receive, Scope, Send

from .httpx_client import custom_httpx_client
from .mcp_server import MCPServerSettings, _global_status, _update_global_activity
from .proxy_server import create_direct_proxy_server, create_guarded_proxy_server

logger = logging.getLogger(__name__)


async def run_http_to_http_proxy(
    target_url: str,
    settings: MCPServerSettings,
    headers: dict[str, Any] | None = None,
    auth: httpx.Auth | None = None,
    verify_ssl: bool | str | None = None,
) -> None:
    """Run an HTTP-to-HTTP proxy with classification.

    Args:
        target_url: The target MCP server URL to proxy to
        settings: Server settings for the proxy
        headers: Headers for connecting to target MCP server
        auth: Optional authentication for the HTTP client
        verify_ssl: Control SSL verification
    """
    logger.info(f"Starting HTTP-to-HTTP proxy from :{settings.port} to {target_url}")

    # Create the target client session
    async with streamablehttp_client(
        url=target_url,
        headers=headers,
        auth=auth,
        httpx_client_factory=partial(custom_httpx_client, verify_ssl=verify_ssl),
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Create the proxy server with classification
            proxy_app = await create_guarded_proxy_server(session)

            # Create the direct proxy server without classification
            direct_proxy_app = await create_direct_proxy_server(session)

            # Set up HTTP server with StreamableHTTP transport for guarded proxy
            guarded_session_manager = StreamableHTTPSessionManager(
                proxy_app,
                stateless=settings.stateless,
            )

            # Set up HTTP server with StreamableHTTP transport for direct proxy
            direct_session_manager = StreamableHTTPSessionManager(
                direct_proxy_app,
                stateless=settings.stateless,
            )

            async def handle_guarded_mcp_request(scope: Scope, receive: Receive, send: Send) -> None:
                await guarded_session_manager.handle_request(scope, receive, send)

            async def handle_direct_mcp_request(scope: Scope, receive: Receive, send: Send) -> None:
                await direct_session_manager.handle_request(scope, receive, send)

            routes: list[BaseRoute] = [
                Mount("/mcp", handle_guarded_mcp_request),
                Mount("/unguarded-mcp", handle_direct_mcp_request),
            ]

            # Add status endpoint
            async def status_endpoint(request: Request) -> JSONResponse:
                _update_global_activity()
                return JSONResponse({
                    "status": "running",
                    "target_url": target_url,
                    "port": settings.port,
                    "api_last_activity": _global_status["api_last_activity"],
                })

            routes.append(Route("/status", status_endpoint, methods=["GET"]))

            middleware = []
            if settings.allow_origins:
                middleware.append(
                    Middleware(
                        CORSMiddleware,
                        allow_origins=settings.allow_origins,
                        allow_credentials=True,
                        allow_methods=["*"],
                        allow_headers=["*"],
                    )
                )

            app = Starlette(
                routes=routes,
                middleware=middleware,
            )

            # Configure uvicorn
            config = uvicorn.Config(
                app,
                host=settings.bind_host,
                port=settings.port,
                log_level=settings.log_level.lower(),
            )

            server = uvicorn.Server(config)

            logger.info(f"HTTP-to-HTTP proxy listening on {settings.bind_host}:{settings.port}")
            logger.info(f"Proxying requests to {target_url}")

            async with guarded_session_manager.run(), direct_session_manager.run():
                await server.serve()
