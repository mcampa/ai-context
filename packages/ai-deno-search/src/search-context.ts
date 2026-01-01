import type {
  Embedding,
  EmbeddingVector,
  HybridSearchRequest,
  HybridSearchResult,
  SemanticSearchResult,
  VectorDatabase,
  VectorSearchResult,
} from "./types.ts";

export interface SearchContextConfig {
  name?: string;
  embedding: Embedding;
  vectorDatabase: VectorDatabase;
  hybridMode?: boolean; // Default true
}

/**
 * SearchContext - Simplified search-only context for Deno
 *
 * This class provides semantic search functionality without indexing capabilities.
 * Use @mcampa/ai-context-core (Node.js) for indexing, then search with this Deno package.
 */
export class SearchContext {
  private name: string;
  private embedding: Embedding;
  private vectorDatabase: VectorDatabase;
  private hybridMode: boolean;

  constructor(config: SearchContextConfig) {
    this.name = config.name || "ai-deno-search";
    this.embedding = config.embedding;
    this.vectorDatabase = config.vectorDatabase;

    // Get from config or env, default to true
    this.hybridMode = config.hybridMode ??
      (Deno.env.get("HYBRID_MODE")?.toLowerCase() === "false" ? false : true);

    console.log(
      `[SearchContext] 🔧 Initialized with hybrid mode: ${this.hybridMode}`,
    );
  }

  /**
   * Generate collection name based on context name and hybrid mode
   */
  private getCollectionName(): string {
    const prefix = this.hybridMode ? "hybrid_code_chunks" : "code_chunks";
    return `${prefix}_${this.name}`;
  }

  /**
   * Perform semantic search on indexed codebase
   * @param query Search query
   * @param topK Number of results to return (default: 5)
   * @param threshold Similarity threshold (default: 0.5)
   * @param filterExpr Optional filter expression
   * @returns Search results
   */
  async semanticSearch(
    query: string,
    topK: number = 5,
    threshold: number = 0.5,
    filterExpr?: string,
  ): Promise<SemanticSearchResult[]> {
    const searchType = this.hybridMode ? "hybrid search" : "semantic search";
    console.log(`[SearchContext] 🔍 Executing ${searchType}: "${query}"`);

    const collectionName = this.getCollectionName();
    console.log(`[SearchContext] 🔍 Using collection: ${collectionName}`);

    // Check if collection exists and has data
    const hasCollection = await this.vectorDatabase.hasCollection(
      collectionName,
    );
    if (!hasCollection) {
      console.log(
        `[SearchContext] ⚠️  Collection '${collectionName}' does not exist. Please index the codebase first using @mcampa/ai-context-core.`,
      );
      return [];
    }

    if (this.hybridMode) {
      try {
        // Check collection stats
        const _stats = await this.vectorDatabase.query(
          collectionName,
          "",
          ["id"],
          1,
        );
        console.log(
          `[SearchContext] 🔍 Collection '${collectionName}' exists and appears to have data`,
        );
      } catch (error) {
        console.log(
          `[SearchContext] ⚠️  Collection '${collectionName}' exists but may be empty or not properly indexed:`,
          error,
        );
      }

      // 1. Generate query vector
      console.log(
        `[SearchContext] 🔍 Generating embeddings for query: "${query}"`,
      );
      const queryEmbedding: EmbeddingVector = await this.embedding.embed(query);
      console.log(
        `[SearchContext] ✅ Generated embedding vector with dimension: ${queryEmbedding.vector.length}`,
      );

      // 2. Prepare hybrid search requests
      const searchRequests: HybridSearchRequest[] = [
        {
          data: queryEmbedding.vector,
          anns_field: "vector",
          param: { nprobe: 10 },
          limit: topK,
        },
        {
          data: query,
          anns_field: "sparse_vector",
          param: { drop_ratio_search: 0.2 },
          limit: topK,
        },
      ];

      // 3. Execute hybrid search
      console.log(
        `[SearchContext] 🔍 Executing hybrid search with RRF reranking...`,
      );
      const searchResults: HybridSearchResult[] = await this.vectorDatabase
        .hybridSearch(collectionName, searchRequests, {
          rerank: {
            strategy: "rrf",
            params: { k: 100 },
          },
          limit: topK,
          filterExpr,
        });

      console.log(
        `[SearchContext] 🔍 Raw search results count: ${searchResults.length}`,
      );

      // 4. Convert to semantic search result format
      const results: SemanticSearchResult[] = searchResults.map((result) => ({
        content: result.document.content,
        relativePath: result.document.relativePath,
        startLine: result.document.startLine,
        endLine: result.document.endLine,
        language: String(result.document.metadata.language || "unknown"),
        score: result.score,
      }));

      console.log(
        `[SearchContext] ✅ Found ${results.length} relevant hybrid results`,
      );
      if (results.length > 0) {
        console.log(
          `[SearchContext] 🔍 Top result score: ${results[0].score}, path: ${
            results[0].relativePath
          }`,
        );
      }

      return results;
    } else {
      // Regular semantic search
      // 1. Generate query vector
      const queryEmbedding: EmbeddingVector = await this.embedding.embed(query);

      // 2. Search in vector database
      const searchResults: VectorSearchResult[] = await this.vectorDatabase
        .search(
          collectionName,
          queryEmbedding.vector,
          { topK, threshold, filterExpr },
        );

      // 3. Convert to semantic search result format
      const results: SemanticSearchResult[] = searchResults.map((result) => ({
        content: result.document.content,
        relativePath: result.document.relativePath,
        startLine: result.document.startLine,
        endLine: result.document.endLine,
        language: String(result.document.metadata.language || "unknown"),
        score: result.score,
      }));

      console.log(
        `[SearchContext] ✅ Found ${results.length} relevant results`,
      );
      return results;
    }
  }

  /**
   * Check if index exists for this context
   * @returns Whether index exists
   */
  async hasIndex(): Promise<boolean> {
    const collectionName = this.getCollectionName();
    return await this.vectorDatabase.hasCollection(collectionName);
  }

  /**
   * Get the embedding provider instance
   */
  getEmbedding(): Embedding {
    return this.embedding;
  }

  /**
   * Get the vector database instance
   */
  getVectorDatabase(): VectorDatabase {
    return this.vectorDatabase;
  }

  /**
   * Get the context name
   */
  getName(): string {
    return this.name;
  }

  /**
   * Get the hybrid mode setting
   */
  isHybridMode(): boolean {
    return this.hybridMode;
  }
}
