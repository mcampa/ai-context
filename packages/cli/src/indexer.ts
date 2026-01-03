import {
  Context,
  MilvusVectorDatabase,
  QdrantVectorDatabase,
  OpenAIEmbedding,
  type VectorDatabase,
} from "@mcampa/ai-context-core";
import type { ContextConfig, IndexResult } from "./types.js";

/**
 * Format a progress bar string
 */
function formatProgressBar(percentage: number, width: number = 30): string {
  const filled = Math.round((percentage / 100) * width);
  const empty = width - filled;
  return `[${"█".repeat(filled)}${"░".repeat(empty)}]`;
}

/**
 * Format duration in human-readable format
 */
function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

/**
 * Run the indexing operation with the provided configuration
 * @param config Validated configuration object
 * @param forceReindex Whether to force reindex even if collection exists
 * @returns Index result with statistics
 */
export async function runIndex(
  config: ContextConfig,
  forceReindex: boolean = false,
): Promise<IndexResult> {
  const startTime = Date.now();
  const cwd = process.cwd();

  console.log("\n🚀 Starting codebase indexing...\n");
  console.log(`📁 Target directory: ${cwd}`);
  console.log(`📛 Context name: ${config.name || "my-context"}`);
  if (forceReindex) {
    console.log(`⚡ Force reindex: enabled`);
  }
  console.log("");

  // Initialize embedding provider
  console.log("🔧 Initializing OpenAI embedding provider...");
  const embedding = new OpenAIEmbedding({
    apiKey: config.embeddingConfig!.apiKey,
    model: config.embeddingConfig!.model,
    baseURL: config.embeddingConfig!.baseURL,
  });
  console.log(`   Model: ${config.embeddingConfig!.model}`);

  // Initialize vector database based on type
  const dbType = config.vectorDatabaseType || "milvus";
  let vectorDatabase: VectorDatabase;

  if (dbType === "qdrant") {
    console.log("🔧 Initializing Qdrant vector database...");
    vectorDatabase = new QdrantVectorDatabase({
      address: config.qdrantConfig!.url,
      apiKey: config.qdrantConfig!.apiKey,
      timeout: config.qdrantConfig!.timeout,
    });
    console.log(`   URL: ${config.qdrantConfig!.url}`);
  } else {
    console.log("🔧 Initializing Milvus vector database...");
    vectorDatabase = new MilvusVectorDatabase({
      address: config.vectorDatabaseConfig!.address,
      token: config.vectorDatabaseConfig!.token,
      username: config.vectorDatabaseConfig!.username,
      password: config.vectorDatabaseConfig!.password,
      ssl: config.vectorDatabaseConfig!.ssl,
    });
  }

  // Create context instance
  console.log("🔧 Creating context instance...\n");
  const context = new Context({
    name: config.name,
    embedding,
    vectorDatabase,
    supportedExtensions: config.supportedExtensions,
    ignorePatterns: config.ignorePatterns,
    customExtensions: config.customExtensions,
    customIgnorePatterns: config.customIgnorePatterns,
  });

  // Track last progress update to avoid console spam
  let lastProgressLine = "";

  // Index the codebase
  const stats = await context.indexCodebase(
    cwd,
    (progress) => {
      const progressLine = `${formatProgressBar(progress.percentage)} ${progress.percentage}% - ${progress.phase}`;

      // Only update if the line changed (to reduce console noise)
      if (progressLine !== lastProgressLine) {
        // Clear previous line and write new one
        process.stdout.write(`\r${progressLine.padEnd(80)}`);
        lastProgressLine = progressLine;
      }
    },
    forceReindex,
  );

  // Clear the progress line and print summary
  process.stdout.write("\r" + " ".repeat(80) + "\r");

  const duration = Date.now() - startTime;

  console.log("\n✅ Indexing completed successfully!\n");
  console.log("📊 Summary:");
  console.log(`   Files indexed: ${stats.indexedFiles}`);
  console.log(`   Total chunks: ${stats.totalChunks}`);
  console.log(`   Status: ${stats.status}`);
  console.log(`   Duration: ${formatDuration(duration)}`);
  console.log("");

  if (stats.status === "limit_reached") {
    console.log(
      "⚠️  Warning: Chunk limit was reached. Some files may not have been fully indexed.",
    );
    console.log("");
  }

  return stats;
}
