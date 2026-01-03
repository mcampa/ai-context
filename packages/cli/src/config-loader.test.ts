import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  existsSync,
  mkdirSync,
  writeFileSync,
  rmSync,
  type PathLike,
} from "fs";
import { join } from "path";
import * as childProcess from "child_process";
import { findConfigFile, loadConfig, findTsxBinary } from "./config-loader.js";

// Mock fs and child_process modules
vi.mock("fs", async () => {
  const actual = await vi.importActual<typeof import("fs")>("fs");
  return {
    ...actual,
    existsSync: vi.fn(actual.existsSync),
  };
});

vi.mock("child_process", async () => {
  const actual =
    await vi.importActual<typeof import("child_process")>("child_process");
  return {
    ...actual,
    execSync: vi.fn(),
    spawnSync: vi.fn(),
  };
});

// Test directory for temporary files
const TEST_DIR = join(process.cwd(), ".test-temp");

// Counter for unique file names to avoid ESM import caching
let testCounter = 0;

function getUniqueConfigPath(ext: string = "js"): string {
  testCounter++;
  return join(TEST_DIR, `test-config-${testCounter}.${ext}`);
}

describe("config-loader", () => {
  beforeEach(() => {
    // Create test directory
    if (!existsSync(TEST_DIR)) {
      mkdirSync(TEST_DIR, { recursive: true });
    }
  });

  afterEach(() => {
    // Clean up test directory
    if (existsSync(TEST_DIR)) {
      rmSync(TEST_DIR, { recursive: true, force: true });
    }
    vi.restoreAllMocks();
    // Restore original environment
    delete process.env.AI_CONTEXT_CLI_ROOT;
  });

  describe("findConfigFile", () => {
    it("should return null when no config file exists", () => {
      const result = findConfigFile(TEST_DIR);
      expect(result).toBeNull();
    });

    it("should find ai-context.config.ts", () => {
      const configPath = join(TEST_DIR, "ai-context.config.ts");
      writeFileSync(configPath, "export default {};");

      const result = findConfigFile(TEST_DIR);
      expect(result).toBe(configPath);
    });

    it("should find ai-context.config.js", () => {
      const configPath = join(TEST_DIR, "ai-context.config.js");
      writeFileSync(configPath, "export default {};");

      const result = findConfigFile(TEST_DIR);
      expect(result).toBe(configPath);
    });

    it("should prefer .ts over .js", () => {
      const tsConfigPath = join(TEST_DIR, "ai-context.config.ts");
      const jsConfigPath = join(TEST_DIR, "ai-context.config.js");
      writeFileSync(tsConfigPath, "export default {};");
      writeFileSync(jsConfigPath, "export default {};");

      const result = findConfigFile(TEST_DIR);
      expect(result).toBe(tsConfigPath);
    });

    it("should find ai-context.config.json", () => {
      const configPath = join(TEST_DIR, "ai-context.config.json");
      writeFileSync(configPath, "{}");

      const result = findConfigFile(TEST_DIR);
      expect(result).toBe(configPath);
    });

    it("should prefer .ts over .json", () => {
      const tsConfigPath = join(TEST_DIR, "ai-context.config.ts");
      const jsonConfigPath = join(TEST_DIR, "ai-context.config.json");
      writeFileSync(tsConfigPath, "export default {};");
      writeFileSync(jsonConfigPath, "{}");

      const result = findConfigFile(TEST_DIR);
      expect(result).toBe(tsConfigPath);
    });

    it("should prefer .js over .json", () => {
      const jsConfigPath = join(TEST_DIR, "ai-context.config.js");
      const jsonConfigPath = join(TEST_DIR, "ai-context.config.json");
      writeFileSync(jsConfigPath, "export default {};");
      writeFileSync(jsonConfigPath, "{}");

      const result = findConfigFile(TEST_DIR);
      expect(result).toBe(jsConfigPath);
    });
  });

  describe("loadConfig", () => {
    it("should throw error when no config file found", async () => {
      const originalCwd = process.cwd();
      process.chdir(TEST_DIR);

      try {
        await expect(loadConfig()).rejects.toThrow("No config file found");
      } finally {
        process.chdir(originalCwd);
      }
    });

    it("should throw error when specified config file not found", async () => {
      await expect(loadConfig("/nonexistent/config.ts")).rejects.toThrow(
        "Config file not found",
      );
    });

    it("should load and validate a valid JavaScript config", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          name: "test-project",
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseConfig: {
            address: "localhost:19530",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      const config = await loadConfig(configPath);

      expect(config.name).toBe("test-project");
      expect(config.embeddingConfig?.apiKey).toBe("test-key");
      expect(config.embeddingConfig?.model).toBe("text-embedding-3-small");
      expect(config.vectorDatabaseConfig?.address).toBe("localhost:19530");
    });

    it("should throw error for missing embeddingConfig", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          vectorDatabaseConfig: {
            address: "localhost:19530",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "Missing required field: embeddingConfig",
      );
    });

    it("should throw error for missing embeddingConfig.apiKey", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            model: "text-embedding-3-small",
          },
          vectorDatabaseConfig: {
            address: "localhost:19530",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "Missing required field: embeddingConfig.apiKey",
      );
    });

    it("should throw error for missing embeddingConfig.model", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
          },
          vectorDatabaseConfig: {
            address: "localhost:19530",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "Missing required field: embeddingConfig.model",
      );
    });

    it("should throw error for missing vectorDatabaseConfig", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "Missing required field: vectorDatabaseConfig",
      );
    });

    it("should throw error when vectorDatabaseConfig has neither address nor token", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseConfig: {
            ssl: true,
          },
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "vectorDatabaseConfig requires either 'address' or 'token' to be set",
      );
    });

    it("should accept config with token instead of address", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseConfig: {
            token: "zilliz-token",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      const config = await loadConfig(configPath);
      expect(config.vectorDatabaseConfig?.token).toBe("zilliz-token");
    });

    it("should throw error for invalid supportedExtensions type", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseConfig: {
            address: "localhost:19530",
          },
          supportedExtensions: "not-an-array",
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "supportedExtensions must be an array of strings",
      );
    });

    it("should throw error for non-string values in supportedExtensions", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseConfig: {
            address: "localhost:19530",
          },
          supportedExtensions: [".ts", 123],
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "supportedExtensions contains non-string value",
      );
    });

    it("should accept valid optional arrays", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseConfig: {
            address: "localhost:19530",
          },
          supportedExtensions: [".ts", ".js"],
          ignorePatterns: ["node_modules/**"],
          customExtensions: [".vue"],
          customIgnorePatterns: ["*.test.ts"],
        };
      `;
      writeFileSync(configPath, configContent);

      const config = await loadConfig(configPath);
      expect(config.supportedExtensions).toEqual([".ts", ".js"]);
      expect(config.ignorePatterns).toEqual(["node_modules/**"]);
      expect(config.customExtensions).toEqual([".vue"]);
      expect(config.customIgnorePatterns).toEqual(["*.test.ts"]);
    });

    it("should load and validate a valid JSON config", async () => {
      const configPath = getUniqueConfigPath("json");
      const configContent = JSON.stringify({
        name: "test-project",
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
        },
        vectorDatabaseConfig: {
          address: "localhost:19530",
        },
      });
      writeFileSync(configPath, configContent);

      const config = await loadConfig(configPath);

      expect(config.name).toBe("test-project");
      expect(config.embeddingConfig?.apiKey).toBe("test-key");
      expect(config.embeddingConfig?.model).toBe("text-embedding-3-small");
      expect(config.vectorDatabaseConfig?.address).toBe("localhost:19530");
    });

    it("should throw error for invalid JSON syntax", async () => {
      const configPath = getUniqueConfigPath("json");
      // Invalid JSON (trailing comma)
      writeFileSync(
        configPath,
        '{ "name": "test", "embeddingConfig": { "apiKey": "key", } }',
      );

      await expect(loadConfig(configPath)).rejects.toThrow("Invalid JSON");
    });

    it("should throw error when JSON config contains null", async () => {
      const configPath = getUniqueConfigPath("json");
      writeFileSync(configPath, "null");

      await expect(loadConfig(configPath)).rejects.toThrow(
        "JSON config must be an object",
      );
    });

    it("should throw error when JSON config contains an array", async () => {
      const configPath = getUniqueConfigPath("json");
      writeFileSync(configPath, "[]");

      await expect(loadConfig(configPath)).rejects.toThrow(
        "JSON config must be an object",
      );
    });

    it("should throw error when JSON config contains a primitive", async () => {
      const configPath = getUniqueConfigPath("json");
      writeFileSync(configPath, '"just a string"');

      await expect(loadConfig(configPath)).rejects.toThrow(
        "JSON config must be an object",
      );
    });

    it("should accept JSON config with all optional fields", async () => {
      const configPath = getUniqueConfigPath("json");
      const configContent = JSON.stringify({
        name: "test-project",
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
          baseURL: "https://custom.api.com",
        },
        vectorDatabaseConfig: {
          address: "localhost:19530",
          ssl: true,
        },
        supportedExtensions: [".py", ".go"],
        ignorePatterns: ["venv/**"],
        customExtensions: [".rs"],
        customIgnorePatterns: ["*.pyc"],
      });
      writeFileSync(configPath, configContent);

      const config = await loadConfig(configPath);
      expect(config.name).toBe("test-project");
      expect(config.embeddingConfig?.baseURL).toBe("https://custom.api.com");
      expect(config.vectorDatabaseConfig?.ssl).toBe(true);
      expect(config.supportedExtensions).toEqual([".py", ".go"]);
      expect(config.customExtensions).toEqual([".rs"]);
    });

    it("should substitute environment variables in JSON config", async () => {
      const configPath = getUniqueConfigPath("json");
      // Set environment variables for the test
      process.env.TEST_API_KEY = "my-secret-api-key";
      process.env.TEST_MILVUS_TOKEN = "my-milvus-token";

      const configContent = JSON.stringify({
        name: "test-project",
        embeddingConfig: {
          apiKey: "[TEST_API_KEY]",
          model: "text-embedding-3-small",
        },
        vectorDatabaseConfig: {
          token: "[TEST_MILVUS_TOKEN]",
          address: "localhost:19530",
        },
      });
      writeFileSync(configPath, configContent);

      try {
        const config = await loadConfig(configPath);
        expect(config.embeddingConfig?.apiKey).toBe("my-secret-api-key");
        expect(config.vectorDatabaseConfig?.token).toBe("my-milvus-token");
      } finally {
        // Clean up environment variables
        delete process.env.TEST_API_KEY;
        delete process.env.TEST_MILVUS_TOKEN;
      }
    });

    it("should throw error for missing environment variable in JSON config", async () => {
      const configPath = getUniqueConfigPath("json");
      // Make sure the env var is not set
      delete process.env.NONEXISTENT_VAR;

      const configContent = JSON.stringify({
        name: "test-project",
        embeddingConfig: {
          apiKey: "[NONEXISTENT_VAR]",
          model: "text-embedding-3-small",
        },
        vectorDatabaseConfig: {
          address: "localhost:19530",
        },
      });
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "Environment variable 'NONEXISTENT_VAR' is not set",
      );
    });

    it("should substitute multiple env vars in the same string", async () => {
      const configPath = getUniqueConfigPath("json");
      process.env.TEST_HOST = "api.example.com";
      process.env.TEST_PORT = "8080";

      const configContent = JSON.stringify({
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
          baseURL: "https://[TEST_HOST]:[TEST_PORT]/v1",
        },
        vectorDatabaseConfig: {
          address: "localhost:19530",
        },
      });
      writeFileSync(configPath, configContent);

      try {
        const config = await loadConfig(configPath);
        expect(config.embeddingConfig?.baseURL).toBe(
          "https://api.example.com:8080/v1",
        );
      } finally {
        delete process.env.TEST_HOST;
        delete process.env.TEST_PORT;
      }
    });

    it("should handle env vars in arrays", async () => {
      const configPath = getUniqueConfigPath("json");
      process.env.TEST_PATTERN = "secret/**";

      const configContent = JSON.stringify({
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
        },
        vectorDatabaseConfig: {
          address: "localhost:19530",
        },
        ignorePatterns: ["node_modules/**", "[TEST_PATTERN]"],
      });
      writeFileSync(configPath, configContent);

      try {
        const config = await loadConfig(configPath);
        expect(config.ignorePatterns).toEqual(["node_modules/**", "secret/**"]);
      } finally {
        delete process.env.TEST_PATTERN;
      }
    });

    it("should load and validate a valid Qdrant config", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          name: "test-project",
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseType: "qdrant",
          qdrantConfig: {
            url: "http://localhost:6333",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      const config = await loadConfig(configPath);

      expect(config.name).toBe("test-project");
      expect(config.vectorDatabaseType).toBe("qdrant");
      expect(config.qdrantConfig?.url).toBe("http://localhost:6333");
    });

    it("should load and validate Qdrant Cloud config with API key", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseType: "qdrant",
          qdrantConfig: {
            url: "https://my-cluster.qdrant.io",
            apiKey: "qdrant-api-key",
            timeout: 15000,
          },
        };
      `;
      writeFileSync(configPath, configContent);

      const config = await loadConfig(configPath);

      expect(config.qdrantConfig?.url).toBe("https://my-cluster.qdrant.io");
      expect(config.qdrantConfig?.apiKey).toBe("qdrant-api-key");
      expect(config.qdrantConfig?.timeout).toBe(15000);
    });

    it("should throw error for missing qdrantConfig when vectorDatabaseType is qdrant", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseType: "qdrant",
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "Missing required field: qdrantConfig",
      );
    });

    it("should throw error for missing qdrantConfig.url", async () => {
      const configPath = getUniqueConfigPath("js");
      const configContent = `
        export default {
          embeddingConfig: {
            apiKey: "test-key",
            model: "text-embedding-3-small",
          },
          vectorDatabaseType: "qdrant",
          qdrantConfig: {
            apiKey: "some-api-key",
          },
        };
      `;
      writeFileSync(configPath, configContent);

      await expect(loadConfig(configPath)).rejects.toThrow(
        "Missing required field: qdrantConfig.url",
      );
    });

    it("should load Qdrant JSON config with environment variable substitution", async () => {
      const configPath = getUniqueConfigPath("json");
      process.env.TEST_QDRANT_URL = "https://my-cluster.qdrant.io";
      process.env.TEST_QDRANT_API_KEY = "my-qdrant-api-key";

      const configContent = JSON.stringify({
        name: "test-project",
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
        },
        vectorDatabaseType: "qdrant",
        qdrantConfig: {
          url: "[TEST_QDRANT_URL]",
          apiKey: "[TEST_QDRANT_API_KEY]",
        },
      });
      writeFileSync(configPath, configContent);

      try {
        const config = await loadConfig(configPath);
        expect(config.qdrantConfig?.url).toBe("https://my-cluster.qdrant.io");
        expect(config.qdrantConfig?.apiKey).toBe("my-qdrant-api-key");
      } finally {
        delete process.env.TEST_QDRANT_URL;
        delete process.env.TEST_QDRANT_API_KEY;
      }
    });
  });

  describe("findTsxBinary", () => {
    const originalEnv = process.env.AI_CONTEXT_CLI_ROOT;
    const originalCwd = process.cwd();

    beforeEach(() => {
      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        // Use actual existsSync for test directory operations
        if (pathStr.includes(".test-temp")) {
          return require("fs").existsSync(path);
        }
        return false;
      });
      vi.mocked(childProcess.execSync).mockClear();
      vi.mocked(childProcess.spawnSync).mockClear();
      delete process.env.AI_CONTEXT_CLI_ROOT;
    });

    afterEach(() => {
      vi.restoreAllMocks();
      if (originalEnv) {
        process.env.AI_CONTEXT_CLI_ROOT = originalEnv;
      }
      process.chdir(originalCwd);
    });

    it("should find tsx in CLI package's node_modules", () => {
      const cliRoot = "/fake/cli/root";
      const tsxPath = join(cliRoot, "node_modules", ".bin", "tsx");
      process.env.AI_CONTEXT_CLI_ROOT = cliRoot;

      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        if (pathStr.includes(".test-temp")) {
          return require("fs").existsSync(path);
        }
        return pathStr === tsxPath;
      });

      const result = findTsxBinary();
      expect(result).toBe(tsxPath);
    });

    it("should find tsx in current working directory's node_modules when CLI package doesn't have it", () => {
      const cwd = TEST_DIR;
      process.chdir(cwd);
      // After chdir, the path will be constructed using process.cwd()
      const expectedPath = join(process.cwd(), "node_modules", ".bin", "tsx");

      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        // Use actual existsSync for test directory operations
        if (pathStr.includes(".test-temp")) {
          const actualFs = require("fs");
          // Only use actual for non-tsx paths
          if (!pathStr.includes("node_modules/.bin/tsx")) {
            return actualFs.existsSync(path);
          }
        }
        // Match the exact path that will be checked
        return pathStr === expectedPath;
      });

      const result = findTsxBinary();
      expect(result).toBe(expectedPath);
    });

    it("should find tsx via which command when not in node_modules", () => {
      const globalTsxPath = "/usr/local/bin/tsx";

      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        if (pathStr.includes(".test-temp")) {
          return require("fs").existsSync(path);
        }
        return pathStr === globalTsxPath;
      });
      vi.mocked(childProcess.execSync).mockReturnValue(globalTsxPath + "\n");

      const result = findTsxBinary();
      expect(result).toBe(globalTsxPath);
    });

    it("should find tsx via direct spawn when available in PATH", () => {
      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        if (pathStr.includes(".test-temp")) {
          return require("fs").existsSync(path);
        }
        return false;
      });
      vi.mocked(childProcess.execSync).mockImplementation(() => {
        throw new Error("tsx not found");
      });
      vi.mocked(childProcess.spawnSync).mockReturnValue({
        status: 0,
        stdout: "",
        stderr: "",
      } as any);

      const result = findTsxBinary();
      expect(result).toBe("tsx");
    });

    it("should return null when tsx is not found anywhere", () => {
      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        if (pathStr.includes(".test-temp")) {
          return require("fs").existsSync(path);
        }
        return false;
      });
      vi.mocked(childProcess.execSync).mockImplementation(() => {
        throw new Error("tsx not found");
      });
      vi.mocked(childProcess.spawnSync).mockReturnValue({
        status: 1,
        stdout: "",
        stderr: "",
      } as any);

      const result = findTsxBinary();
      expect(result).toBeNull();
    });

    it("should prefer CLI package's tsx over local node_modules", () => {
      const cliRoot = "/fake/cli/root";
      const cliTsxPath = join(cliRoot, "node_modules", ".bin", "tsx");
      const cwd = TEST_DIR;
      const localTsxPath = join(cwd, "node_modules", ".bin", "tsx");
      process.env.AI_CONTEXT_CLI_ROOT = cliRoot;
      process.chdir(cwd);

      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        if (pathStr.includes(".test-temp")) {
          return require("fs").existsSync(path);
        }
        return pathStr === cliTsxPath || pathStr === localTsxPath;
      });

      const result = findTsxBinary();
      expect(result).toBe(cliTsxPath);
    });

    it("should prefer local node_modules over global tsx", () => {
      const cwd = TEST_DIR;
      process.chdir(cwd);
      // After chdir, the path will be constructed using process.cwd()
      const expectedLocalPath = join(
        process.cwd(),
        "node_modules",
        ".bin",
        "tsx",
      );
      const globalTsxPath = "/usr/local/bin/tsx";

      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        // Use actual existsSync for test directory operations
        if (pathStr.includes(".test-temp")) {
          const actualFs = require("fs");
          // Only use actual for non-tsx paths
          if (!pathStr.includes("node_modules/.bin/tsx")) {
            return actualFs.existsSync(path);
          }
        }
        // Match the exact paths that will be checked
        return pathStr === expectedLocalPath || pathStr === globalTsxPath;
      });
      vi.mocked(childProcess.execSync).mockReturnValue(globalTsxPath + "\n");

      const result = findTsxBinary();
      expect(result).toBe(expectedLocalPath);
    });

    it("should handle Windows which command (where)", () => {
      const originalPlatform = process.platform;
      Object.defineProperty(process, "platform", {
        value: "win32",
        writable: true,
      });

      const globalTsxPath = "C:\\Program Files\\nodejs\\tsx.cmd";

      vi.mocked(existsSync).mockImplementation((path: PathLike) => {
        const pathStr = String(path);
        if (pathStr.includes(".test-temp")) {
          return require("fs").existsSync(path);
        }
        return pathStr === globalTsxPath;
      });
      vi.mocked(childProcess.execSync).mockReturnValue(globalTsxPath + "\r\n");

      const result = findTsxBinary();
      expect(result).toBe(globalTsxPath);

      Object.defineProperty(process, "platform", {
        value: originalPlatform,
        writable: true,
      });
    });
  });
});
