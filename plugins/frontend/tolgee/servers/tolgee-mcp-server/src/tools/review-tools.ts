/**
 * Review and comment tools for Tolgee MCP server
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getTolgeeClient } from "../services/tolgee-client.js";
import {
  ListCommentsSchema,
  AddCommentSchema,
  ResolveCommentSchema,
  type ListCommentsInput,
  type AddCommentInput,
  type ResolveCommentInput,
} from "../schemas/index.js";
import { formatComments, formatComment } from "./formatters.js";

export function registerReviewTools(server: McpServer): void {
  const client = getTolgeeClient();

  // List comments
  server.registerTool(
    "list_comments",
    {
      title: "List Translation Comments",
      description:
        "List all comments on a specific translation. Comments are used for collaboration and review feedback.",
      inputSchema: ListCommentsSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: ListCommentsInput) => {
      try {
        const result = await client.listComments(params.translation_id);
        return {
          content: [
            {
              type: "text",
              text: formatComments(result, params.translation_id, params.response_format),
            },
          ],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error listing comments: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Add comment
  server.registerTool(
    "add_comment",
    {
      title: "Add Translation Comment",
      description:
        "Add a comment to a translation for review feedback or collaboration. Use NEEDS_RESOLUTION state to flag issues that need attention.",
      inputSchema: AddCommentSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async (params: AddCommentInput) => {
      try {
        const result = await client.addComment(params.translation_id, {
          text: params.text,
          state: params.state,
        });
        return {
          content: [{ type: "text", text: formatComment(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error adding comment: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Resolve comment
  server.registerTool(
    "resolve_comment",
    {
      title: "Resolve Translation Comment",
      description:
        "Mark a comment as resolved. Use this after addressing the feedback or issue raised in the comment.",
      inputSchema: ResolveCommentSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: ResolveCommentInput) => {
      try {
        await client.setCommentState(params.translation_id, params.comment_id, "RESOLVED");
        return {
          content: [
            {
              type: "text",
              text: `Successfully resolved comment ${params.comment_id} on translation ${params.translation_id}`,
            },
          ],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error resolving comment: ${(error as Error).message}` }],
        };
      }
    }
  );
}
