import { describe, it, expect } from "vitest";
import type {
  ContextConfig,
  OpenAIEmbeddingConfig,
  MilvusConfig,
  QdrantConfig,
  VectorDatabaseType,
  CLIOptions,
  IndexResult,
} from "./types.js";

describe("Types", () => {
  describe("ContextConfig", () => {
    it("should allow a minimal valid config", () => {
      const config: ContextConfig = {
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
        },
        vectorDatabaseConfig: {
          address: "localhost:19530",
        },
      };

      expect(config.embeddingConfig?.apiKey).toBe("test-key");
      expect(config.embeddingConfig?.model).toBe("text-embedding-3-small");
      expect(config.vectorDatabaseConfig?.address).toBe("localhost:19530");
    });

    it("should allow all optional fields", () => {
      const config: ContextConfig = {
        name: "my-project",
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
          baseURL: "https://custom.api.com",
        },
        vectorDatabaseConfig: {
          address: "localhost:19530",
          token: "token",
          username: "user",
          password: "pass",
          ssl: true,
        },
        supportedExtensions: [".ts", ".js"],
        ignorePatterns: ["node_modules/**"],
        customExtensions: [".vue"],
        customIgnorePatterns: ["*.test.ts"],
      };

      expect(config.name).toBe("my-project");
      expect(config.embeddingConfig?.baseURL).toBe("https://custom.api.com");
      expect(config.vectorDatabaseConfig?.ssl).toBe(true);
      expect(config.supportedExtensions).toHaveLength(2);
      expect(config.customExtensions).toContain(".vue");
    });

    it("should allow config with only token for vector database", () => {
      const config: ContextConfig = {
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
        },
        vectorDatabaseConfig: {
          token: "zilliz-cloud-token",
        },
      };

      expect(config.vectorDatabaseConfig?.token).toBe("zilliz-cloud-token");
      expect(config.vectorDatabaseConfig?.address).toBeUndefined();
    });
  });

  describe("OpenAIEmbeddingConfig", () => {
    it("should require apiKey and model", () => {
      const config: OpenAIEmbeddingConfig = {
        apiKey: "sk-test",
        model: "text-embedding-3-small",
      };

      expect(config.apiKey).toBe("sk-test");
      expect(config.model).toBe("text-embedding-3-small");
    });

    it("should allow optional baseURL", () => {
      const config: OpenAIEmbeddingConfig = {
        apiKey: "sk-test",
        model: "text-embedding-3-small",
        baseURL: "https://api.example.com/v1",
      };

      expect(config.baseURL).toBe("https://api.example.com/v1");
    });
  });

  describe("MilvusConfig", () => {
    it("should allow address-based config", () => {
      const config: MilvusConfig = {
        address: "localhost:19530",
      };

      expect(config.address).toBe("localhost:19530");
    });

    it("should allow token-based config", () => {
      const config: MilvusConfig = {
        token: "my-token",
        ssl: true,
      };

      expect(config.token).toBe("my-token");
      expect(config.ssl).toBe(true);
    });

    it("should allow username/password config", () => {
      const config: MilvusConfig = {
        address: "localhost:19530",
        username: "root",
        password: "milvus",
      };

      expect(config.username).toBe("root");
      expect(config.password).toBe("milvus");
    });
  });

  describe("QdrantConfig", () => {
    it("should require url", () => {
      const config: QdrantConfig = {
        url: "http://localhost:6333",
      };

      expect(config.url).toBe("http://localhost:6333");
    });

    it("should allow api key for cloud instances", () => {
      const config: QdrantConfig = {
        url: "https://my-cluster.qdrant.io",
        apiKey: "my-api-key",
      };

      expect(config.url).toBe("https://my-cluster.qdrant.io");
      expect(config.apiKey).toBe("my-api-key");
    });

    it("should allow optional timeout", () => {
      const config: QdrantConfig = {
        url: "http://localhost:6333",
        timeout: 30000,
      };

      expect(config.timeout).toBe(30000);
    });
  });

  describe("ContextConfig with Qdrant", () => {
    it("should allow Qdrant configuration", () => {
      const config: ContextConfig = {
        name: "my-project",
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
        },
        vectorDatabaseType: "qdrant",
        qdrantConfig: {
          url: "http://localhost:6333",
        },
      };

      expect(config.vectorDatabaseType).toBe("qdrant");
      expect(config.qdrantConfig?.url).toBe("http://localhost:6333");
    });

    it("should allow Qdrant Cloud configuration", () => {
      const config: ContextConfig = {
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

      expect(config.qdrantConfig?.apiKey).toBe("qdrant-api-key");
      expect(config.qdrantConfig?.timeout).toBe(15000);
    });

    it("should allow explicit milvus type", () => {
      const dbType: VectorDatabaseType = "milvus";
      const config: ContextConfig = {
        embeddingConfig: {
          apiKey: "test-key",
          model: "text-embedding-3-small",
        },
        vectorDatabaseType: dbType,
        vectorDatabaseConfig: {
          address: "localhost:19530",
        },
      };

      expect(config.vectorDatabaseType).toBe("milvus");
    });
  });

  describe("CLIOptions", () => {
    it("should allow empty options", () => {
      const options: CLIOptions = {};

      expect(options.config).toBeUndefined();
      expect(options.force).toBeUndefined();
    });

    it("should allow all options", () => {
      const options: CLIOptions = {
        config: "./custom-config.ts",
        force: true,
      };

      expect(options.config).toBe("./custom-config.ts");
      expect(options.force).toBe(true);
    });
  });

  describe("IndexResult", () => {
    it("should represent completed indexing", () => {
      const result: IndexResult = {
        indexedFiles: 100,
        totalChunks: 500,
        status: "completed",
      };

      expect(result.indexedFiles).toBe(100);
      expect(result.totalChunks).toBe(500);
      expect(result.status).toBe("completed");
    });

    it("should represent limit_reached status", () => {
      const result: IndexResult = {
        indexedFiles: 50,
        totalChunks: 450000,
        status: "limit_reached",
      };

      expect(result.status).toBe("limit_reached");
    });
  });
});
