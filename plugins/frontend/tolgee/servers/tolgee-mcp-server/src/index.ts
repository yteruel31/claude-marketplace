/**
 * Tolgee MCP Server entry point
 *
 * Provides MCP tools for managing translations in Tolgee.
 * Supports stdio transport for Claude Desktop integration.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import express from "express";

import { getTolgeeClient } from "./services/tolgee-client.js";
import { registerKeyTools } from "./tools/key-tools.js";
import { registerTranslationTools } from "./tools/translation-tools.js";
import { registerReviewTools } from "./tools/review-tools.js";
import { registerLanguageTools } from "./tools/language-tools.js";
import { registerTaskTools } from "./tools/task-tools.js";

const SERVER_NAME = "tolgee-mcp-server";
const SERVER_VERSION = "1.0.0";

function createServer(): McpServer {
  const server = new McpServer({
    name: SERVER_NAME,
    version: SERVER_VERSION,
  });

  // Initialize and validate client (throws if API key missing)
  const client = getTolgeeClient();
  console.error(`[${SERVER_NAME}] Connected to Tolgee project ${client.getProjectId()} at ${client.getHost()}`);

  // Register all tools
  registerKeyTools(server);
  registerTranslationTools(server);
  registerReviewTools(server);
  registerLanguageTools(server);
  registerTaskTools(server);

  console.error(`[${SERVER_NAME}] Registered 20 tools`);

  return server;
}

async function runStdio(): Promise<void> {
  const server = createServer();
  const transport = new StdioServerTransport();

  await server.connect(transport);
  console.error(`[${SERVER_NAME}] Server running on stdio`);
}

async function runHttp(port: number): Promise<void> {
  const app = express();

  app.get("/health", (_req, res) => {
    res.json({ status: "ok", server: SERVER_NAME, version: SERVER_VERSION });
  });

  app.get("/", (_req, res) => {
    res.json({
      name: SERVER_NAME,
      version: SERVER_VERSION,
      description: "Tolgee MCP Server for translation management",
      usage: "Connect via MCP protocol or use /health for status",
    });
  });

  app.listen(port, () => {
    console.error(`[${SERVER_NAME}] HTTP server listening on port ${port}`);
    console.error(`[${SERVER_NAME}] Health check: http://localhost:${port}/health`);
  });
}

function printHelp(): void {
  console.log(`
Tolgee MCP Server v${SERVER_VERSION}

Usage: node dist/index.js [options]

Options:
  --stdio     Run in stdio mode (default, for Claude Desktop)
  --http      Run in HTTP mode
  --help      Show this help message

Environment Variables:
  TOLGEE_API_KEY      Required. Your Tolgee project API key (tgpak_...)
  TOLGEE_PROJECT_ID   Required. Tolgee project ID
  TOLGEE_HOST         Optional. Tolgee instance URL (default: https://app.tolgee.io)
  PORT                Optional. HTTP server port (default: 3000)

Examples:
  # Run in stdio mode (for Claude Desktop)
  TOLGEE_API_KEY=tgpak_... TOLGEE_PROJECT_ID=12345 node dist/index.js --stdio

  # Run in HTTP mode
  TOLGEE_API_KEY=tgpak_... TOLGEE_PROJECT_ID=12345 node dist/index.js --http
`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.includes("--help") || args.includes("-h")) {
    printHelp();
    process.exit(0);
  }

  try {
    if (args.includes("--http")) {
      const port = parseInt(process.env.PORT || "3000");
      await runHttp(port);
    } else {
      // Default to stdio mode
      await runStdio();
    }
  } catch (error) {
    console.error(`[${SERVER_NAME}] Fatal error: ${(error as Error).message}`);
    process.exit(1);
  }
}

main();
