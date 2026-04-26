/**
 * Task management tools for Tolgee MCP server
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getTolgeeClient } from "../services/tolgee-client.js";
import {
  ListTasksSchema,
  GetTaskSchema,
  CreateTaskSchema,
  UpdateTaskSchema,
  TaskNumberSchema,
  type ListTasksInput,
  type GetTaskInput,
  type CreateTaskInput,
  type UpdateTaskInput,
  type TaskNumberInput,
} from "../schemas/index.js";
import { formatTaskList, formatTask } from "./formatters.js";

export function registerTaskTools(server: McpServer): void {
  const client = getTolgeeClient();

  // List tasks
  server.registerTool(
    "list_tasks",
    {
      title: "List Tasks",
      description:
        "List tasks in the Tolgee project with optional filtering by state, type, or assignee. Returns paginated results.",
      inputSchema: ListTasksSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: ListTasksInput) => {
      try {
        const result = await client.listTasks({
          state: params.state,
          type: params.type,
          assigneeId: params.assignee_id,
          page: params.page,
          size: params.size,
        });
        return {
          content: [{ type: "text", text: formatTaskList(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error listing tasks: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Get task details
  server.registerTool(
    "get_task",
    {
      title: "Get Task Details",
      description:
        "Get detailed information about a specific task, including progress, assignees, and metadata.",
      inputSchema: GetTaskSchema,
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: GetTaskInput) => {
      try {
        const result = await client.getTask(params.task_number);
        return {
          content: [{ type: "text", text: formatTask(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error getting task: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Create task
  server.registerTool(
    "create_task",
    {
      title: "Create Task",
      description:
        "Create a new translation or review task. Requires a task type (TRANSLATE or REVIEW) and target language ID.",
      inputSchema: CreateTaskSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async (params: CreateTaskInput) => {
      try {
        const result = await client.createTask({
          name: params.name,
          description: params.description || "", // required field, default to empty string
          type: params.type,
          languageId: params.language_id,
          dueDate: params.due_date ? new Date(params.due_date).getTime() : undefined,
          assignees: params.assignee_ids || [], // required field, default to empty array
          keys: params.key_ids || [], // required field, default to empty array
        });
        return {
          content: [{ type: "text", text: formatTask(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error creating task: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Update task
  server.registerTool(
    "update_task",
    {
      title: "Update Task",
      description:
        "Update an existing task. Can modify the name, description, due date, and assignees.",
      inputSchema: UpdateTaskSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
    },
    async (params: UpdateTaskInput) => {
      try {
        const result = await client.updateTask(params.task_number, {
          name: params.name,
          description: params.description,
          dueDate: params.due_date === null ? null : params.due_date ? new Date(params.due_date).getTime() : undefined,
          assignees: params.assignee_ids,
        });
        return {
          content: [{ type: "text", text: formatTask(result, params.response_format) }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error updating task: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Finish task
  server.registerTool(
    "finish_task",
    {
      title: "Finish Task",
      description:
        "Mark a task as finished (DONE state). Use this when all translation/review work is complete.",
      inputSchema: TaskNumberSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async (params: TaskNumberInput) => {
      try {
        await client.finishTask(params.task_number);
        return {
          content: [{ type: "text", text: `Task #${params.task_number} has been marked as finished.` }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error finishing task: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Close task
  server.registerTool(
    "close_task",
    {
      title: "Close Task",
      description:
        "Close a task (CLOSED state). Use this to archive a task that is no longer needed or was cancelled.",
      inputSchema: TaskNumberSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async (params: TaskNumberInput) => {
      try {
        await client.closeTask(params.task_number);
        return {
          content: [{ type: "text", text: `Task #${params.task_number} has been closed.` }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error closing task: ${(error as Error).message}` }],
        };
      }
    }
  );

  // Reopen task
  server.registerTool(
    "reopen_task",
    {
      title: "Reopen Task",
      description:
        "Reopen a previously closed or finished task. Use this to resume work on a task.",
      inputSchema: TaskNumberSchema,
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async (params: TaskNumberInput) => {
      try {
        await client.reopenTask(params.task_number);
        return {
          content: [{ type: "text", text: `Task #${params.task_number} has been reopened.` }],
        };
      } catch (error) {
        return {
          isError: true,
          content: [{ type: "text", text: `Error reopening task: ${(error as Error).message}` }],
        };
      }
    }
  );
}
