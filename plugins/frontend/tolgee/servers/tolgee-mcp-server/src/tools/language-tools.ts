/**
 * Language and namespace tools for Tolgee MCP server
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getTolgeeClient } from "../services/tolgee-client.js";
import {
  ListLanguagesSchema,
  ListNamespacesSchema,
  type ListLanguagesInput,
  type ListNamespacesInput,
} from "../schemas/index.js";
import { formatLanguages, formatNamespaces } from "./formatters.js";

export function registerLanguageTools(server: McpServer): void {
  const client = getTolgeeClient();

  // List languages
  server.registerTool(
    "list_languages",
    {
      title: "List Project Languages",
      description:
        "List all languages configured in the Tolgee project. Shows language codes, names, and which is the base language.",
      inputSchema: ListLanguagesSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: ListLanguagesInput) => {
      try {
        const result = await client.listLanguages();
        return {
          content: [{ type: "text", text: formatLanguages(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error listing languages: ${(error as Error).message}` }],
        };
      }
    }
  );

  // List namespaces
  server.registerTool(
    "list_namespaces",
    {
      title: "List Namespaces",
      description:
        "List all namespaces in the Tolgee project with their key counts. Namespaces help organize translations by feature or domain.",
      inputSchema: ListNamespacesSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: ListNamespacesInput) => {
      try {
        const result = await client.listNamespaces();
        return {
          content: [{ type: "text", text: formatNamespaces(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error listing namespaces: ${(error as Error).message}` }],
        };
      }
    }
  );
}
