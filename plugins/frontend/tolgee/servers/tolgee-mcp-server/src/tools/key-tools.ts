/**
 * Key management tools for Tolgee MCP server
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getTolgeeClient } from "../services/tolgee-client.js";
import {
  ListKeysSchema,
  GetKeySchema,
  CreateKeySchema,
  UpdateKeySchema,
  DeleteKeysSchema,
  type ListKeysInput,
  type GetKeyInput,
  type CreateKeyInput,
  type UpdateKeyInput,
  type DeleteKeysInput,
} from "../schemas/index.js";
import { formatKeysList, formatKey, formatTranslationsResponse } from "./formatters.js";

export function registerKeyTools(server: McpServer): void {
  const client = getTolgeeClient();

  // List keys
  server.registerTool(
    "list_keys",
    {
      title: "List Translation Keys",
      description:
        "List translation keys in the Tolgee project with optional filtering by namespace, search term, tags, translation state, languages, and more. Returns paginated results with translations. Supports sorting.",
      inputSchema: ListKeysSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: ListKeysInput) => {
      try {
        const result = await client.listKeys({
          namespace: params.namespace,
          search: params.search,
          tags: params.tags,
          filterState: params.filter_state,
          languages: params.languages,
          filterUntranslatedInLang: params.filter_untranslated_in_lang,
          filterTranslatedInLang: params.filter_translated_in_lang,
          filterHasScreenshot: params.filter_has_screenshot,
          filterKeyName: params.filter_key_name,
          sort: params.sort,
          page: params.page,
          size: params.size,
        });
        return {
          content: [{ type: "text", text: formatKeysList(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error listing keys: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Get key details
  server.registerTool(
    "get_key",
    {
      title: "Get Translation Key Details",
      description:
        "Get detailed information about a specific translation key, including all translations, tags, screenshots, and metadata.",
      inputSchema: GetKeySchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: GetKeyInput) => {
      try {
        const result = await client.getKey(params.key_id);
        return {
          content: [{ type: "text", text: formatKey(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error getting key: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Create key
  server.registerTool(
    "create_key",
    {
      title: "Create Translation Key",
      description:
        "Create a new translation key with optional initial translations. Use dot notation for key names (e.g., 'checkout.pay_button').",
      inputSchema: CreateKeySchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async (params: CreateKeyInput) => {
      try {
        const result = await client.createKey({
          key: params.key_name,
          namespace: params.namespace || undefined,
          translations: params.translations,
          tags: params.tags,
        });
        return {
          content: [{ type: "text", text: formatTranslationsResponse(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error creating key: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Update key
  server.registerTool(
    "update_key",
    {
      title: "Update Translation Key",
      description:
        "Update an existing translation key. Can modify the key name, namespace, description, tags, and translations all in one operation.",
      inputSchema: UpdateKeySchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: UpdateKeyInput) => {
      try {
        const result = await client.updateKey(params.key_id, {
          name: params.name,
          namespace: params.namespace,
          description: params.description,
          tags: params.tags,
          translations: params.translations,
        });
        return {
          content: [{ type: "text", text: formatKey(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error updating key: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Delete keys
  server.registerTool(
    "delete_keys",
    {
      title: "Delete Translation Keys",
      description:
        "Permanently delete one or more translation keys. This action cannot be undone - all translations for the deleted keys will be lost.",
      inputSchema: DeleteKeysSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: true,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: DeleteKeysInput) => {
      try {
        await client.deleteKeys(params.key_ids);
        return {
          content: [
            {
              type: "text",
              text: `Successfully deleted ${params.key_ids.length} key(s): ${params.key_ids.join(", ")}`,
            },
          ],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error deleting keys: ${(error as Error).message}` }],
        };
      }
    }
  );
}
