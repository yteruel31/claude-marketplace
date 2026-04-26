/**
 * Tolgee API type definitions
 */

import type { TranslationState, CommentState, TaskType, TaskState } from "./constants.js";

// Pagination
export interface PagedResponse<T> {
  _embedded?: {
    keys?: T[];
    languages?: T[];
    namespaces?: T[];
  };
  page?: {
    size: number;
    number: number;
    totalElements: number;
    totalPages: number;
  };
}

// Language
export interface Language {
  id: number;
  name: string;
  tag: string;
  originalName: string;
  flagEmoji: string | null;
  base: boolean;
}

// Namespace
export interface Namespace {
  id: number;
  name: string;
}

export interface UsedNamespace {
  name: string | null;
  keyCount: number;
}

// Key
export interface Key {
  id: number;
  name: string;
  namespace: string | null;
  description: string | null;
  tags: Tag[];
  isPlural: boolean;
}

export interface KeyWithTranslations extends Key {
  translations: Record<string, Translation>;
  screenshots: Screenshot[];
}

// Tag
export interface Tag {
  id: number;
  name: string;
}

// Translation
export interface Translation {
  id: number;
  text: string | null;
  state: TranslationState;
  outdated: boolean;
  auto: boolean;
  mtProvider: string | null;
}

// Comment
export interface Comment {
  id: number;
  text: string;
  state: CommentState;
  author: User;
  createdAt: string;
  updatedAt: string;
}

// User
export interface User {
  id: number;
  username: string;
  name: string | null;
  avatarHash: string | null;
}

// Screenshot
export interface Screenshot {
  id: number;
  filename: string;
  thumbnail: string;
  width: number;
  height: number;
}

// Request/Response types for specific endpoints

// Create key request
export interface CreateKeyRequest {
  key: string;
  namespace?: string | null;
  translations?: Record<string, string>;
  tags?: string[];
}

// Update key request (complex update)
export interface UpdateKeyRequest {
  name?: string;
  namespace?: string | null;
  description?: string | null;
  tags?: string[];
  translations?: Record<string, string>;
}

// Set translation request
export interface SetTranslationRequest {
  key: string;
  namespace?: string | null;
  translations: Record<string, string>;
}

// Create comment request
export interface CreateCommentRequest {
  text: string;
  state?: CommentState;
}

// Keys list response
export interface KeysListResponse {
  _embedded?: {
    keys?: KeyWithTranslations[];
  };
  page?: {
    size: number;
    number: number;
    totalElements: number;
    totalPages: number;
  };
}

// Translations response (for key creation)
export interface TranslationsResponse {
  keyId: number;
  keyName: string;
  keyNamespace: string | null;
  translations: Record<string, Translation>;
}

// ==================== Tasks ====================

// Task response model (matches Tolgee API response)
export interface Task {
  number: number;
  name: string | null;
  description: string;
  type: TaskType;
  language: Language;
  dueDate: number | null; // timestamp
  assignees: User[];
  state: TaskState;
  author: User | null;
  totalItems: number;
  doneItems: number;
  baseWordCount: number;
  baseCharacterCount: number;
  createdAt: number;
  closedAt: number | null;
}

// Create task request (all fields with * are required by the API)
export interface CreateTaskRequest {
  name?: string; // optional, 3-255 characters
  description: string; // required, 0-2000 characters
  type: TaskType; // required: TRANSLATE or REVIEW
  languageId: number; // required
  dueDate?: number; // optional, timestamp
  assignees: number[]; // required - user IDs (can be empty array)
  keys: number[]; // required - key IDs to include (can be empty array)
}

// Update task request
export interface UpdateTaskRequest {
  name?: string;
  description?: string;
  dueDate?: number | null;
  assignees?: number[];
}

// Tasks list response
export interface TasksListResponse {
  _embedded?: { tasks?: Task[] };
  page?: {
    size: number;
    number: number;
    totalElements: number;
    totalPages: number;
  };
}
