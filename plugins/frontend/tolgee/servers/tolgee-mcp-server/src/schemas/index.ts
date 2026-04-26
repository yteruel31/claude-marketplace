/**
 * Zod schemas for Tolgee MCP tools input validation
 */

import { z } from "zod";
import { TRANSLATION_STATES, FILTER_TRANSLATION_STATES, COMMENT_STATES, RESPONSE_FORMATS, TASK_TYPES, TASK_STATES } from "../constants.js";

// Common schemas
export const ResponseFormatSchema = z
  .enum(RESPONSE_FORMATS)
  .default("markdown")
  .describe("Output format: 'json' for raw data or 'markdown' for human-readable");

export const PaginationSchema = z.object({
  page: z.number().int().min(0).default(0).describe("Page number (0-indexed)"),
  size: z.number().int().min(1).max(100).default(20).describe("Number of items per page (max 100)"),
});

// ==================== Key Schemas ====================

export const ListKeysSchema = z
  .object({
    namespace: z.string().optional().describe("Filter by namespace"),
    search: z.string().optional().describe("Search term for key names and translation values"),
    tags: z.array(z.string()).optional().describe("Filter by tags"),
    filter_state: z
      .array(z.enum(FILTER_TRANSLATION_STATES))
      .optional()
      .describe("Filter by translation state: UNTRANSLATED, TRANSLATED, REVIEWED, or DISABLED"),
    languages: z
      .array(z.string())
      .optional()
      .describe("Language codes to limit which translations are returned per key (e.g., ['en', 'sv']). Does not filter keys - only controls which language translations appear in the response"),
    filter_untranslated_in_lang: z
      .array(z.string())
      .optional()
      .describe("Filter keys missing translations in these languages. Note: using the same language in both filter_untranslated_in_lang and filter_translated_in_lang returns empty results"),
    filter_translated_in_lang: z
      .array(z.string())
      .optional()
      .describe("Filter keys that have translations in these languages. Note: using the same language in both filter_translated_in_lang and filter_untranslated_in_lang returns empty results"),
    filter_has_screenshot: z
      .boolean()
      .optional()
      .describe("Filter keys with (true) or without (false) screenshots"),
    filter_key_name: z
      .string()
      .optional()
      .describe("Filter keys whose name contains this string"),
    sort: z
      .array(z.string())
      .optional()
      .describe("Sort criteria. Valid fields: 'id', 'name', 'createdAt', 'updatedAt'. Directions: 'asc', 'desc'. Format: ['name,asc', 'createdAt,desc']"),
    page: z.number().int().min(0).default(0).describe("Page number (0-indexed)"),
    size: z.number().int().min(1).max(100).default(20).describe("Items per page"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type ListKeysInput = z.infer<typeof ListKeysSchema>;

export const GetKeySchema = z
  .object({
    key_id: z.number().int().positive().describe("The ID of the key to retrieve"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type GetKeyInput = z.infer<typeof GetKeySchema>;

export const CreateKeySchema = z
  .object({
    key_name: z
      .string()
      .min(1)
      .max(2000)
      .describe("Unique key name, e.g., 'checkout.pay_button' or 'homepage.welcome_message'"),
    namespace: z.string().nullable().optional().describe("Optional namespace for organization"),
    translations: z
      .record(z.string())
      .optional()
      .describe("Initial translations by language code, e.g., { 'en': 'Pay Now', 'sv': 'Betala nu' }"),
    tags: z.array(z.string()).optional().describe("Tags to apply to the key"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type CreateKeyInput = z.infer<typeof CreateKeySchema>;

export const UpdateKeySchema = z
  .object({
    key_id: z.number().int().positive().describe("The ID of the key to update"),
    name: z.string().min(1).max(2000).optional().describe("New name for the key"),
    namespace: z.string().nullable().optional().describe("New namespace (null to remove)"),
    description: z.string().nullable().optional().describe("Description of the key"),
    tags: z.array(z.string()).optional().describe("Tags to set on the key (replaces existing)"),
    translations: z
      .record(z.string())
      .optional()
      .describe("Translations to update by language code"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type UpdateKeyInput = z.infer<typeof UpdateKeySchema>;

export const DeleteKeysSchema = z
  .object({
    key_ids: z.array(z.number().int().positive()).min(1).describe("Array of key IDs to delete"),
  })
  .strict();

export type DeleteKeysInput = z.infer<typeof DeleteKeysSchema>;

// ==================== Translation Schemas ====================

export const SetTranslationSchema = z
  .object({
    key_name: z.string().min(1).describe("The key name to set translation for"),
    namespace: z.string().nullable().optional().describe("Namespace of the key (if applicable)"),
    translations: z
      .record(z.string())
      .describe("Translations by language code, e.g., { 'en': 'Hello', 'sv': 'Hej' }"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type SetTranslationInput = z.infer<typeof SetTranslationSchema>;

export const SetTranslationStateSchema = z
  .object({
    translation_id: z.number().int().positive().describe("The translation ID to update"),
    state: z.enum(TRANSLATION_STATES).describe("New state: UNTRANSLATED, TRANSLATED, or REVIEWED"),
  })
  .strict();

export type SetTranslationStateInput = z.infer<typeof SetTranslationStateSchema>;

export const MarkOutdatedSchema = z
  .object({
    translation_id: z.number().int().positive().describe("The translation ID to mark"),
    outdated: z.boolean().describe("Whether to mark the translation as outdated"),
  })
  .strict();

export type MarkOutdatedInput = z.infer<typeof MarkOutdatedSchema>;

// ==================== Comment Schemas ====================

export const ListCommentsSchema = z
  .object({
    translation_id: z.number().int().positive().describe("The translation ID to get comments for"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type ListCommentsInput = z.infer<typeof ListCommentsSchema>;

export const AddCommentSchema = z
  .object({
    translation_id: z.number().int().positive().describe("The translation ID to add comment to"),
    text: z.string().min(1).max(10000).describe("Comment text (supports Markdown)"),
    state: z
      .enum(COMMENT_STATES)
      .default("RESOLUTION_NOT_NEEDED")
      .describe("Comment state: RESOLUTION_NOT_NEEDED, NEEDS_RESOLUTION, or RESOLVED"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type AddCommentInput = z.infer<typeof AddCommentSchema>;

export const ResolveCommentSchema = z
  .object({
    translation_id: z.number().int().positive().describe("The translation ID the comment belongs to"),
    comment_id: z.number().int().positive().describe("The comment ID to resolve"),
  })
  .strict();

export type ResolveCommentInput = z.infer<typeof ResolveCommentSchema>;

// ==================== Language Schemas ====================

export const ListLanguagesSchema = z
  .object({
    response_format: ResponseFormatSchema,
  })
  .strict();

export type ListLanguagesInput = z.infer<typeof ListLanguagesSchema>;

export const ListNamespacesSchema = z
  .object({
    response_format: ResponseFormatSchema,
  })
  .strict();

export type ListNamespacesInput = z.infer<typeof ListNamespacesSchema>;

// ==================== Task Schemas ====================

export const ListTasksSchema = z
  .object({
    state: z.enum(TASK_STATES).optional().describe("Filter by task state: NEW, IN_PROGRESS, DONE, or CLOSED"),
    type: z.enum(TASK_TYPES).optional().describe("Filter by task type: TRANSLATE or REVIEW"),
    assignee_id: z.number().int().positive().optional().describe("Filter by assignee user ID"),
    page: z.number().int().min(0).default(0).describe("Page number (0-indexed)"),
    size: z.number().int().min(1).max(100).default(20).describe("Items per page"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type ListTasksInput = z.infer<typeof ListTasksSchema>;

export const GetTaskSchema = z
  .object({
    task_number: z.number().int().positive().describe("The task number"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type GetTaskInput = z.infer<typeof GetTaskSchema>;

export const CreateTaskSchema = z
  .object({
    name: z.string().max(255).optional().describe("Task name (max 255 characters)"),
    description: z.string().max(2000).optional().describe("Task description (max 2000 characters)"),
    type: z.enum(TASK_TYPES).describe("Task type: TRANSLATE or REVIEW"),
    language_id: z.number().int().positive().describe("Target language ID"),
    due_date: z.string().optional().describe("Due date in ISO format (e.g., '2024-12-31')"),
    assignee_ids: z.array(z.number().int().positive()).optional().describe("User IDs to assign"),
    key_ids: z.array(z.number().int().positive()).optional().describe("Translation key IDs to include"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type CreateTaskInput = z.infer<typeof CreateTaskSchema>;

export const UpdateTaskSchema = z
  .object({
    task_number: z.number().int().positive().describe("The task number to update"),
    name: z.string().max(255).optional().describe("New task name"),
    description: z.string().max(2000).optional().describe("New description"),
    due_date: z.string().nullable().optional().describe("New due date in ISO format (null to clear)"),
    assignee_ids: z.array(z.number().int().positive()).optional().describe("New assignee user IDs"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type UpdateTaskInput = z.infer<typeof UpdateTaskSchema>;

export const TaskNumberSchema = z
  .object({
    task_number: z.number().int().positive().describe("The task number"),
    response_format: ResponseFormatSchema,
  })
  .strict();

export type TaskNumberInput = z.infer<typeof TaskNumberSchema>;
