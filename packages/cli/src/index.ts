import { Command } from "commander";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { loadConfig } from "./config-loader.js";
import { runIndex } from "./indexer.js";
import type { CLIOptions } from "./types.js";

// Re-export types for consumers
export type {
  ContextConfig,
  OpenAIEmbeddingConfig,
  MilvusConfig,
  CLIOptions,
  IndexResult,
} from "./types.js";

// Read version from package.json to keep it in sync
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const packageJson = JSON.parse(
  readFileSync(join(__dirname, "..", "package.json"), "utf-8"),
);
const VERSION = packageJson.version;

const program = new Command();

program
  .name("ai-context-index")
  .description("Index your codebase for AI-powered semantic search")
  .version(VERSION, "-v, --version", "Display version number")
  .option(
    "-c, --config <path>",
    "Path to config file (default: ai-context.config.ts/js)",
  )
  .option("-f, --force", "Force reindex even if collection already exists")
  .action(async (options: CLIOptions) => {
    try {
      console.log("");
      console.log(
        "╔════════════════════════════════════════════════════════════╗",
      );
      console.log(
        "║             AI Context CLI - Codebase Indexer              ║",
      );
      console.log(`║                        Version ${VERSION.padEnd(25)} ║`);
      console.log(
        "╚════════════════════════════════════════════════════════════╝",
      );
      console.log("");

      // Load and validate config
      const config = await loadConfig(options.config);

      // Run the indexing operation
      await runIndex(config, options.force);

      process.exit(0);
    } catch (error) {
      console.error("");
      console.error(
        "❌ Error:",
        error instanceof Error ? error.message : String(error),
      );
      console.error("");

      if (error instanceof Error && error.stack && process.env.DEBUG) {
        console.error("Stack trace:");
        console.error(error.stack);
        console.error("");
      }

      process.exit(1);
    }
  });

// Handle uncaught errors
process.on("uncaughtException", (error) => {
  console.error("");
  console.error("❌ Uncaught error:", error.message);
  if (process.env.DEBUG) {
    console.error(error.stack);
  }
  process.exit(1);
});

process.on("unhandledRejection", (reason) => {
  console.error("");
  console.error("❌ Unhandled rejection:", reason);
  process.exit(1);
});

// Show version if no arguments provided (just the command name)
if (process.argv.length === 2) {
  console.log(`@mcampa/ai-context-cli version ${VERSION}`);
  process.exit(0);
}

program.parse();
