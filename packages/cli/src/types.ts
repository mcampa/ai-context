/**
 * Configuration for OpenAI embedding provider
 */
export interface OpenAIEmbeddingConfig {
  /** The embedding model to use (e.g., "text-embedding-3-small") */
  model: string;
  /** Your OpenAI API key */
  apiKey: string;
  /** Optional custom base URL for OpenAI-compatible APIs */
  baseURL?: string;
}

/**
 * Configuration for Milvus vector database
 */
export interface MilvusConfig {
  /** Milvus server address (e.g., "localhost:19530") */
  address?: string;
  /** Authentication token (for Zilliz Cloud) */
  token?: string;
  /** Username for authentication */
  username?: string;
  /** Password for authentication */
  password?: string;
  /** Whether to use SSL connection */
  ssl?: boolean;
}

/**
 * Configuration for Qdrant vector database
 */
export interface QdrantConfig {
  /** Qdrant server URL (e.g., "http://localhost:6333" or "https://your-cluster.qdrant.io") */
  url: string;
  /** API key for Qdrant Cloud (optional for self-hosted instances) */
  apiKey?: string;
  /** Connection timeout in milliseconds (default: 10000) */
  timeout?: number;
}

/**
 * Supported vector database types
 */
export type VectorDatabaseType = "milvus" | "qdrant";

/**
 * Main configuration interface for ai-context.config.ts/js
 */
export interface ContextConfig {
  /** Name identifier for this context (used in collection naming) */
  name?: string;
  /** OpenAI embedding configuration */
  embeddingConfig?: OpenAIEmbeddingConfig;
  /**
   * Vector database type to use
   * @default "milvus"
   */
  vectorDatabaseType?: VectorDatabaseType;
  /** Milvus vector database configuration (used when vectorDatabaseType is "milvus" or not set) */
  vectorDatabaseConfig?: MilvusConfig;
  /** Qdrant vector database configuration (used when vectorDatabaseType is "qdrant") */
  qdrantConfig?: QdrantConfig;
  /**
   * Supported file extensions for indexing
   * If provided, replaces the default extensions
   */
  supportedExtensions?: string[];
  /**
   * Patterns to ignore during indexing
   * If provided, replaces the default ignore patterns
   */
  ignorePatterns?: string[];
  /**
   * Additional file extensions to include beyond defaults
   * Merged with default or provided supportedExtensions
   */
  customExtensions?: string[];
  /**
   * Additional patterns to ignore beyond defaults
   * Merged with default or provided ignorePatterns
   */
  customIgnorePatterns?: string[];
}

/**
 * CLI options parsed from command line arguments
 */
export interface CLIOptions {
  /** Path to config file (optional, defaults to ai-context.config.ts/js in cwd) */
  config?: string;
  /** Force reindex even if collection exists */
  force?: boolean;
}

/**
 * Result of a successful index operation
 */
export interface IndexResult {
  /** Number of files indexed */
  indexedFiles: number;
  /** Total number of chunks created */
  totalChunks: number;
  /** Status of the indexing operation */
  status: "completed" | "limit_reached";
}
