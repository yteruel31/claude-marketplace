/**
 * Translation management tools for Tolgee MCP server
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getTolgeeClient } from "../services/tolgee-client.js";
import {
  SetTranslationSchema,
  SetTranslationStateSchema,
  MarkOutdatedSchema,
  type SetTranslationInput,
  type SetTranslationStateInput,
  type MarkOutdatedInput,
} from "../schemas/index.js";
import { formatTranslationsResponse } from "./formatters.js";

export function registerTranslationTools(server: McpServer): void {
  const client = getTolgeeClient();

  // Set translation
  server.registerTool(
    "set_translation",
    {
      title: "Set Translation",
      description:
        "Set or update the translation text for a key in one or more languages. Creates the key if it doesn't exist.",
      inputSchema: SetTranslationSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: SetTranslationInput) => {
      try {
        const result = await client.setTranslation({
          key: params.key_name,
          namespace: params.namespace || undefined,
          translations: params.translations,
        });
        return {
          content: [{ type: "text", text: formatTranslationsResponse(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error setting translation: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Set translation state
  server.registerTool(
    "set_translation_state",
    {
      title: "Set Translation State",
      description:
        "Change the state of a translation. States are: UNTRANSLATED (not yet translated), TRANSLATED (translated but not reviewed), REVIEWED (approved).",
      inputSchema: SetTranslationStateSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: SetTranslationStateInput) => {
      try {
        await client.setTranslationState(params.translation_id, params.state);
        return {
          content: [
            {
              type: "text",
              text: `Successfully set translation ${params.translation_id} state to ${params.state}`,
            },
          ],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error setting state: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Mark outdated
  server.registerTool(
    "mark_outdated",
    {
      title: "Mark Translation Outdated",
      description:
        "Mark a translation as outdated (needs re-translation) or not outdated. Use this when the source text changes and translations need updating.",
      inputSchema: MarkOutdatedSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: MarkOutdatedInput) => {
      try {
        await client.setTranslationOutdated(params.translation_id, params.outdated);
        const action = params.outdated ? "marked as outdated" : "marked as up-to-date";
        return {
          content: [
            {
              type: "text",
              text: `Successfully ${action} translation ${params.translation_id}`,
            },
          ],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error marking outdated: ${(error as Error).message}` }],
        };
      }
    }
  );
}
