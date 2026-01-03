import type { Context } from "../../src/context";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FakeEmbedding } from "../doubles/fake-embedding";
import { FakeVectorDatabase } from "../doubles/fake-vector-database";
import { TestContextBuilder } from "../doubles/test-context-builder";

describe("lifecycle Integration", () => {
  let context: Context;
  let fakeDb: FakeVectorDatabase;
  let fakeEmbedding: FakeEmbedding;
  let fixturesPath: string;

  // Helper function to create context with specific name
  const createContext = (name: string = "my-context") => {
    return new TestContextBuilder()
      .withName(name)
      .withEmbedding(fakeEmbedding)
      .withVectorDatabase(fakeDb)
      .build();
  };

  beforeEach(() => {
    // Create test doubles
    fakeDb = new FakeVectorDatabase({ address: "test" });
    fakeEmbedding = new FakeEmbedding(128);

    // Create context with test doubles
    context = createContext();

    // Path to test fixtures
    fixturesPath = path.join(__dirname, "../fixtures/sample-codebase");
  });

  afterEach(() => {
    // Clean up test doubles
    fakeDb.reset();
    fakeEmbedding.reset();
  });

  describe("collection Creation", () => {
    it("should create collection during first index", async () => {
      // Arrange
      const collectionName = context.getCollectionName();

      // Assert: Collection does not exist initially
      expect(await fakeDb.hasCollection(collectionName)).toBe(false);

      // Act
      await context.indexCodebase(fixturesPath);

      // Assert: Collection was created
      expect(await fakeDb.hasCollection(collectionName)).toBe(true);
    });

    it("should use existing collection on subsequent index", async () => {
      // Arrange: Index once
      await context.indexCodebase(fixturesPath);

      const collectionName = context.getCollectionName();
      const firstDocCount = fakeDb.getCollectionDocumentCount(collectionName);

      // Act: Index again (without force)
      await context.indexCodebase(fixturesPath);

      // Assert: Document count should increase (cumulative)
      const secondDocCount = fakeDb.getCollectionDocumentCount(collectionName);
      expect(secondDocCount).toBeGreaterThanOrEqual(firstDocCount);
    });

    it("should generate consistent collection name for same context", () => {
      // Act
      const name1 = context.getCollectionName();
      const name2 = context.getCollectionName();

      // Assert
      expect(name1).toBe(name2);
    });

    it("should generate different collection names for different context names", () => {
      // Arrange: Create contexts with different names
      const context1 = createContext("codebase1");
      const context2 = createContext("codebase2");

      // Act
      const name1 = context1.getCollectionName();
      const name2 = context2.getCollectionName();

      // Assert
      expect(name1).not.toBe(name2);
    });
  });

  describe("index Clearing", () => {
    it("should remove collection when clearing index", async () => {
      // Arrange: Index codebase
      await context.indexCodebase(fixturesPath);

      const collectionName = context.getCollectionName();

      // Assert: Collection exists
      expect(await fakeDb.hasCollection(collectionName)).toBe(true);

      // Act: Clear index
      await context.clearIndex(fixturesPath);

      // Assert: Collection was removed
      expect(await fakeDb.hasCollection(collectionName)).toBe(false);
    });

    it("should remove all documents when clearing index", async () => {
      // Arrange: Index codebase
      await context.indexCodebase(fixturesPath);

      const collectionName = context.getCollectionName();
      const docCountBefore = fakeDb.getCollectionDocumentCount(collectionName);

      expect(docCountBefore).toBeGreaterThan(0);

      // Act: Clear index
      await context.clearIndex(fixturesPath);

      // Assert: No documents remain
      expect(await fakeDb.hasCollection(collectionName)).toBe(false);
    });

    it("should handle clearing non-existent index gracefully", async () => {
      // Arrange: Non-existent codebase path
      const nonExistentPath = "/path/to/nonexistent";

      // Act & Assert: Should not throw
      await expect(context.clearIndex(nonExistentPath)).resolves.not.toThrow();
    });

    it("should allow re-indexing after clearing", async () => {
      // Arrange: Index, then clear
      await context.indexCodebase(fixturesPath);
      await context.clearIndex(fixturesPath);

      const collectionName = context.getCollectionName();

      // Assert: Collection was removed
      expect(await fakeDb.hasCollection(collectionName)).toBe(false);

      // Act: Re-index
      const result = await context.indexCodebase(fixturesPath);

      // Assert: Collection was recreated
      expect(await fakeDb.hasCollection(collectionName)).toBe(true);
      expect(result.indexedFiles).toBeGreaterThan(0);
      expect(result.totalChunks).toBeGreaterThan(0);
    });
  });

  describe("multiple Codebases", () => {
    it("should manage multiple codebases independently with different contexts", async () => {
      // Arrange: Create separate contexts for different codebases
      const context1 = createContext("codebase-1");
      const context2 = createContext("codebase-2");
      const fixturesPath2 = path.join(
        __dirname,
        "../fixtures/sample-codebase-2",
      );

      // Act: Index both codebases with their respective contexts
      await context1.indexCodebase(fixturesPath);
      await context2.indexCodebase(fixturesPath2);

      // Assert: Both collections exist
      const collection1 = context1.getCollectionName();
      const collection2 = context2.getCollectionName();

      expect(await fakeDb.hasCollection(collection1)).toBe(true);
      expect(await fakeDb.hasCollection(collection2)).toBe(true);

      // Verify collections are different
      expect(collection1).not.toBe(collection2);
    });

    it("should clear one codebase without affecting others", async () => {
      // Arrange: Create separate contexts and index two codebases
      const context1 = createContext("codebase-1");
      const context2 = createContext("codebase-2");
      const fixturesPath2 = path.join(
        __dirname,
        "../fixtures/sample-codebase-2",
      );

      await context1.indexCodebase(fixturesPath);
      await context2.indexCodebase(fixturesPath2);

      const collection1 = context1.getCollectionName();
      const collection2 = context2.getCollectionName();

      // Act: Clear first codebase
      await context1.clearIndex(fixturesPath);

      // Assert: First collection removed, second remains
      expect(await fakeDb.hasCollection(collection1)).toBe(false);
      expect(await fakeDb.hasCollection(collection2)).toBe(true);
    });

    it("should search each codebase independently", async () => {
      // Arrange: Index codebase
      await context.indexCodebase(fixturesPath);

      // Act: Search
      const results = await context.semanticSearch("user", 5);

      // Assert: Results are from the correct codebase
      expect(results.length).toBeGreaterThan(0);

      // All results should be from fixtures path
      for (const result of results) {
        expect(result.relativePath).toBeDefined();
      }
    });
  });

  describe("index Status", () => {
    it("should indicate collection exists after indexing", async () => {
      // Arrange
      const collectionName = context.getCollectionName();

      // Assert: Initially does not exist
      expect(await fakeDb.hasCollection(collectionName)).toBe(false);

      // Act
      await context.indexCodebase(fixturesPath);

      // Assert: Now exists
      expect(await fakeDb.hasCollection(collectionName)).toBe(true);
    });

    it("should indicate collection does not exist before indexing", async () => {
      // Arrange
      const collectionName = context.getCollectionName();

      // Act & Assert
      expect(await fakeDb.hasCollection(collectionName)).toBe(false);
    });

    it("should list all indexed collections", async () => {
      // Arrange: Create contexts for different codebases
      const context1 = createContext("codebase-1");
      const context2 = createContext("codebase-2");
      const fixturesPath2 = path.join(
        __dirname,
        "../fixtures/sample-codebase-2",
      );

      await context1.indexCodebase(fixturesPath);
      await context2.indexCodebase(fixturesPath2);

      // Act
      const collections = await fakeDb.listCollections();

      // Assert
      expect(collections.length).toBe(2);

      const expectedCollections = [
        context1.getCollectionName(),
        context2.getCollectionName(),
      ];

      for (const expected of expectedCollections) {
        expect(collections).toContain(expected);
      }
    });
  });

  describe("complete Workflow", () => {
    it("should complete full lifecycle: index → search → clear → verify", async () => {
      // Step 1: Index
      const indexResult = await context.indexCodebase(fixturesPath);
      expect(indexResult.indexedFiles).toBeGreaterThan(0);

      const collectionName = context.getCollectionName();
      expect(await fakeDb.hasCollection(collectionName)).toBe(true);

      // Step 2: Search
      const searchResults = await context.semanticSearch("user", 5);
      expect(searchResults.length).toBeGreaterThan(0);

      // Step 3: Clear
      await context.clearIndex(fixturesPath);
      expect(await fakeDb.hasCollection(collectionName)).toBe(false);

      // Step 4: Verify search returns empty (no collection)
      const searchAfterClear = await context.semanticSearch("user", 5);
      expect(searchAfterClear).toEqual([]);
    });

    it("should handle full lifecycle multiple times", async () => {
      // Cycle 1
      await context.indexCodebase(fixturesPath);
      const results1 = await context.semanticSearch("user", 5);
      await context.clearIndex(fixturesPath);

      // Cycle 2
      await context.indexCodebase(fixturesPath);
      const results2 = await context.semanticSearch("user", 5);
      await context.clearIndex(fixturesPath);

      // Assert: Both cycles worked
      expect(results1.length).toBeGreaterThan(0);
      expect(results2.length).toBeGreaterThan(0);
    });
  });

  describe("error Handling", () => {
    it("should handle database errors during clear", async () => {
      // Arrange: Index first
      await context.indexCodebase(fixturesPath);

      // Inject failure
      fakeDb.injectFailure();

      // Act & Assert
      await expect(context.clearIndex(fixturesPath)).rejects.toThrow();
    });
  });
});
