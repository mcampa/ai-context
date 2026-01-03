import { existsSync, readFileSync } from "fs";
import { resolve, join } from "path";
import { pathToFileURL } from "url";
import { spawnSync, execSync } from "child_process";
import type { ContextConfig } from "./types.js";

const CONFIG_FILE_NAMES = [
  "ai-context.config.ts",
  "ai-context.config.js",
  "ai-context.config.json",
];

/**
 * Find the config file in the given directory
 * Searches for ai-context.config.ts first, then ai-context.config.js
 */
export function findConfigFile(cwd: string): string | null {
  for (const fileName of CONFIG_FILE_NAMES) {
    const filePath = join(cwd, fileName);
    if (existsSync(filePath)) {
      return filePath;
    }
  }
  return null;
}

/**
 * Find tsx binary in multiple locations
 */
export function findTsxBinary(): string | null {
  // 1. Check CLI package's node_modules
  const cliRoot = process.env.AI_CONTEXT_CLI_ROOT;
  if (cliRoot) {
    const cliPath = join(cliRoot, "node_modules", ".bin", "tsx");
    if (existsSync(cliPath)) {
      return cliPath;
    }
  }

  // 2. Check current working directory's node_modules (where user is running the command)
  const cwd = process.cwd();
  const localPath = join(cwd, "node_modules", ".bin", "tsx");
  if (existsSync(localPath)) {
    return localPath;
  }

  // 3. Check if tsx is available in PATH (globally installed)
  try {
    // Use 'which' on Unix-like systems, 'where' on Windows
    const command = process.platform === "win32" ? "where tsx" : "which tsx";
    const tsxPath = execSync(command, { encoding: "utf-8", stdio: "pipe" })
      .trim()
      .split("\n")[0];
    if (tsxPath && existsSync(tsxPath)) {
      return tsxPath;
    }
  } catch {
    // tsx not found in PATH, continue to next check
  }

  // 4. Try to use 'tsx' directly (might work if it's in PATH and spawn can find it)
  // We'll test this by trying to spawn it
  try {
    const testResult = spawnSync("tsx", ["--version"], {
      encoding: "utf-8",
      stdio: "pipe",
    });
    if (testResult.status === 0) {
      return "tsx"; // Return the command name if it works
    }
  } catch {
    // tsx command not available
  }

  return null;
}

/**
 * Load a TypeScript config file by spawning tsx
 */
function loadTypeScriptConfig(filePath: string): ContextConfig {
  // Find tsx binary in multiple locations
  const tsxPath = findTsxBinary();

  if (!tsxPath) {
    throw new Error(
      "TypeScript config files require tsx to be installed.\n" +
        "Please use a JavaScript config file (ai-context.config.js) instead,\n" +
        "or install tsx: npm install -g tsx",
    );
  }

  // Create a wrapper script to load and export the config
  const wrapperScript = `
    import config from ${JSON.stringify(pathToFileURL(filePath).href)};
    console.log(JSON.stringify(config.default || config));
  `;

  const result = spawnSync(tsxPath, ["--eval", wrapperScript], {
    encoding: "utf-8",
    cwd: process.cwd(),
    env: {
      ...process.env,
      // Don't inherit NODE_PATH to avoid resolution issues
      NODE_PATH: undefined,
    },
  });

  if (result.error) {
    throw new Error(`Failed to run tsx: ${result.error.message}`);
  }

  if (result.status !== 0) {
    const errorOutput = result.stderr || result.stdout || "Unknown error";
    throw new Error(`Failed to load TypeScript config:\n${errorOutput}`);
  }

  try {
    return JSON.parse(result.stdout.trim());
  } catch {
    throw new Error(
      `Failed to parse config output. Make sure your config exports a valid object.\n` +
        `Output was: ${result.stdout}`,
    );
  }
}

/**
 * Load a JavaScript config file
 */
async function loadJavaScriptConfig(filePath: string): Promise<ContextConfig> {
  const fileUrl = pathToFileURL(filePath).href;
  const module = await import(fileUrl);
  return module.default || module;
}

/**
 * Replace environment variable placeholders in a string
 * Placeholders are in the format [ENV_VAR_NAME]
 */
function substituteEnvVars(value: string): string {
  return value.replace(/\[([A-Z_][A-Z0-9_]*)\]/g, (match, envVarName) => {
    const envValue = process.env[envVarName];
    if (envValue === undefined) {
      throw new Error(
        `Environment variable '${envVarName}' is not set (referenced as ${match} in config)`,
      );
    }
    return envValue;
  });
}

/**
 * Recursively process an object and substitute environment variables in string values
 */
function processEnvVars<T>(obj: T): T {
  if (typeof obj === "string") {
    return substituteEnvVars(obj) as T;
  }
  if (Array.isArray(obj)) {
    return obj.map((item) => processEnvVars(item)) as T;
  }
  if (obj !== null && typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      result[key] = processEnvVars(value);
    }
    return result as T;
  }
  return obj;
}

/**
 * Load a JSON config file
 * Supports environment variable substitution using [ENV_VAR_NAME] syntax
 */
function loadJsonConfig(filePath: string): ContextConfig {
  try {
    const content = readFileSync(filePath, "utf-8");
    const parsed = JSON.parse(content);

    // Validate that parsed JSON is an object (not null, array, or primitive)
    if (
      parsed === null ||
      typeof parsed !== "object" ||
      Array.isArray(parsed)
    ) {
      throw new Error(
        "JSON config must be an object, not " +
          (parsed === null
            ? "null"
            : Array.isArray(parsed)
              ? "an array"
              : typeof parsed),
      );
    }

    return processEnvVars(parsed);
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(`Invalid JSON in config file: ${error.message}`);
    }
    throw error;
  }
}

/**
 * Load and validate the configuration file
 * @param configPath Path to the config file, or directory to search for config
 * @returns Validated configuration object
 */
export async function loadConfig(configPath?: string): Promise<ContextConfig> {
  const cwd = process.cwd();
  let resolvedPath: string;

  if (configPath) {
    // User provided a specific config path
    resolvedPath = resolve(cwd, configPath);
    if (!existsSync(resolvedPath)) {
      throw new Error(`Config file not found: ${resolvedPath}`);
    }
  } else {
    // Search for config file in cwd
    const foundPath = findConfigFile(cwd);
    if (!foundPath) {
      throw new Error(
        `No config file found. Please create one of:\n` +
          CONFIG_FILE_NAMES.map((f) => `  - ${f}`).join("\n") +
          `\n\nExample JS/TS config (Milvus):\n\n` +
          `export default {\n` +
          `  name: "my-project",\n` +
          `  embeddingConfig: {\n` +
          `    apiKey: process.env.OPENAI_API_KEY,\n` +
          `    model: "text-embedding-3-small",\n` +
          `  },\n` +
          `  vectorDatabaseConfig: {\n` +
          `    address: "localhost:19530",\n` +
          `  },\n` +
          `};\n\n` +
          `Example JS/TS config (Qdrant):\n\n` +
          `export default {\n` +
          `  name: "my-project",\n` +
          `  embeddingConfig: {\n` +
          `    apiKey: process.env.OPENAI_API_KEY,\n` +
          `    model: "text-embedding-3-small",\n` +
          `  },\n` +
          `  vectorDatabaseType: "qdrant",\n` +
          `  qdrantConfig: {\n` +
          `    url: "http://localhost:6333",\n` +
          `  },\n` +
          `};\n\n` +
          `Example JSON config (ai-context.config.json):\n\n` +
          `{\n` +
          `  "name": "my-project",\n` +
          `  "embeddingConfig": {\n` +
          `    "apiKey": "[OPENAI_API_KEY]",\n` +
          `    "model": "text-embedding-3-small"\n` +
          `  },\n` +
          `  "vectorDatabaseConfig": {\n` +
          `    "address": "localhost:19530"\n` +
          `  }\n` +
          `}`,
      );
    }
    resolvedPath = foundPath;
  }

  console.log(`📄 Loading config from: ${resolvedPath}`);

  try {
    let config: ContextConfig;

    if (resolvedPath.endsWith(".ts") || resolvedPath.endsWith(".tsx")) {
      config = loadTypeScriptConfig(resolvedPath);
    } else if (resolvedPath.endsWith(".json")) {
      config = loadJsonConfig(resolvedPath);
    } else {
      config = await loadJavaScriptConfig(resolvedPath);
    }

    // Validate the config
    validateConfig(config);

    return config;
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Failed to load config: ${error.message}`);
    }
    throw error;
  }
}

/**
 * Validate the configuration object
 * @param config Configuration to validate
 * @throws Error if validation fails
 */
function validateConfig(config: ContextConfig): void {
  const errors: string[] = [];

  // Check required embedding config
  if (!config.embeddingConfig) {
    errors.push("Missing required field: embeddingConfig");
  } else {
    if (!config.embeddingConfig.apiKey) {
      errors.push("Missing required field: embeddingConfig.apiKey");
    }
    if (!config.embeddingConfig.model) {
      errors.push("Missing required field: embeddingConfig.model");
    }
  }

  // Check required vector database config based on type
  const dbType = config.vectorDatabaseType || "milvus";

  if (dbType === "qdrant") {
    // Validate Qdrant configuration
    if (!config.qdrantConfig) {
      errors.push(
        "Missing required field: qdrantConfig (required when vectorDatabaseType is 'qdrant')",
      );
    } else {
      if (!config.qdrantConfig.url) {
        errors.push("Missing required field: qdrantConfig.url");
      }
    }
  } else {
    // Validate Milvus configuration (default)
    if (!config.vectorDatabaseConfig) {
      errors.push("Missing required field: vectorDatabaseConfig");
    } else {
      // At least address or token is required
      if (
        !config.vectorDatabaseConfig.address &&
        !config.vectorDatabaseConfig.token
      ) {
        errors.push(
          "vectorDatabaseConfig requires either 'address' or 'token' to be set",
        );
      }
    }
  }

  // Validate supportedExtensions format if provided
  if (config.supportedExtensions) {
    if (!Array.isArray(config.supportedExtensions)) {
      errors.push("supportedExtensions must be an array of strings");
    } else {
      for (const ext of config.supportedExtensions) {
        if (typeof ext !== "string") {
          errors.push(`supportedExtensions contains non-string value: ${ext}`);
        }
      }
    }
  }

  // Validate ignorePatterns format if provided
  if (config.ignorePatterns) {
    if (!Array.isArray(config.ignorePatterns)) {
      errors.push("ignorePatterns must be an array of strings");
    }
  }

  // Validate customExtensions format if provided
  if (config.customExtensions) {
    if (!Array.isArray(config.customExtensions)) {
      errors.push("customExtensions must be an array of strings");
    }
  }

  // Validate customIgnorePatterns format if provided
  if (config.customIgnorePatterns) {
    if (!Array.isArray(config.customIgnorePatterns)) {
      errors.push("customIgnorePatterns must be an array of strings");
    }
  }

  if (errors.length > 0) {
    throw new Error(
      `Invalid configuration:\n${errors.map((e) => `  - ${e}`).join("\n")}`,
    );
  }

  console.log("✅ Config validated successfully");
}
