/**
 * Tolgee MCP Server constants
 */

// Default configuration
export const DEFAULT_HOST = "https://app.tolgee.io";

// API limits
export const DEFAULT_PAGE_SIZE = 20;
export const MAX_PAGE_SIZE = 100;
export const API_TIMEOUT = 30000; // 30 seconds

// Translation states
export const TRANSLATION_STATES = ["UNTRANSLATED", "TRANSLATED", "REVIEWED"] as const;
export type TranslationState = (typeof TRANSLATION_STATES)[number];

// Translation states for filtering (includes DISABLED which is only valid as a filter, not for set-state)
export const FILTER_TRANSLATION_STATES = ["UNTRANSLATED", "TRANSLATED", "REVIEWED", "DISABLED"] as const;
export type FilterTranslationState = (typeof FILTER_TRANSLATION_STATES)[number];

// Comment states
export const COMMENT_STATES = ["RESOLUTION_NOT_NEEDED", "NEEDS_RESOLUTION", "RESOLVED"] as const;
export type CommentState = (typeof COMMENT_STATES)[number];

// Task types
export const TASK_TYPES = ["TRANSLATE", "REVIEW"] as const;
export type TaskType = (typeof TASK_TYPES)[number];

// Task states
export const TASK_STATES = ["NEW", "IN_PROGRESS", "DONE", "CLOSED"] as const;
export type TaskState = (typeof TASK_STATES)[number];

// Response format options
export const RESPONSE_FORMATS = ["json", "markdown"] as const;
export type ResponseFormat = (typeof RESPONSE_FORMATS)[number];

// Output limits
export const MAX_OUTPUT_LENGTH = 50000; // 50KB max output
