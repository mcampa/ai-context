/**
 * Basic Search Example for Deno
 *
 * This example demonstrates how to use the SearchContext to perform
 * semantic search on a codebase that has been indexed using @mcampa/ai-context-core.
 *
 * Prerequisites:
 * 1. Index your codebase using Node.js core package
 * 2. Set environment variables: OPENAI_API_KEY, MILVUS_ADDRESS, MILVUS_TOKEN
 *
 * Run with:
 * deno run --allow-env --allow-net examples/basic-search.ts
 */

import {
  MilvusRestfulVectorDatabase,
  OpenAIEmbedding,
  SearchContext,
} from "../mod.ts";

// Configuration
const CONTEXT_NAME = "my-project"; // Must match the name used during indexing
const SEARCH_QUERY = "function that handles user authentication";
const TOP_K = 5;

async function main() {
  console.log("🚀 Starting Deno Search Example\n");

  // 1. Check environment variables
  const openaiKey = Deno.env.get("OPENAI_API_KEY");
  const milvusAddress = Deno.env.get("MILVUS_ADDRESS");
  const milvusToken = Deno.env.get("MILVUS_TOKEN");

  if (!openaiKey || !milvusAddress || !milvusToken) {
    console.error("❌ Missing required environment variables:");
    console.error("   OPENAI_API_KEY:", openaiKey ? "✓" : "✗");
    console.error("   MILVUS_ADDRESS:", milvusAddress ? "✓" : "✗");
    console.error("   MILVUS_TOKEN:", milvusToken ? "✓" : "✗");
    Deno.exit(1);
  }

  // 2. Initialize embedding provider
  console.log("🔧 Initializing OpenAI embedding provider...");
  const embedding = new OpenAIEmbedding({
    apiKey: openaiKey,
    model: "text-embedding-3-small",
  });

  // 3. Initialize vector database (uses REST API)
  console.log("🔧 Connecting to Milvus vector database...");
  const vectorDatabase = new MilvusRestfulVectorDatabase({
    address: milvusAddress,
    token: milvusToken,
  });

  // 4. Create search context
  // IMPORTANT: Use the same name as used during indexing with core package
  console.log(`🔧 Creating search context: "${CONTEXT_NAME}"\n`);
  const searchContext = new SearchContext({
    name: CONTEXT_NAME,
    embedding,
    vectorDatabase,
  });

  // 5. Check if index exists
  console.log("🔍 Checking if index exists...");
  const hasIndex = await searchContext.hasIndex();

  if (!hasIndex) {
    console.error(`\n❌ No index found for context "${CONTEXT_NAME}".\n`);
    console.error(
      "Please index your codebase first using @mcampa/ai-context-core:\n",
    );
    console.error("  import { Context } from '@mcampa/ai-context-core';");
    console.error(
      "  const context = new Context({ name: 'my-project', ... });",
    );
    console.error("  await context.indexCodebase('./path/to/codebase');\n");
    Deno.exit(1);
  }

  console.log("✅ Index found!\n");

  // 6. Perform semantic search
  console.log(`🔍 Searching for: "${SEARCH_QUERY}"`);
  console.log(`   Top K: ${TOP_K}\n`);

  const startTime = performance.now();
  const results = await searchContext.semanticSearch(SEARCH_QUERY, TOP_K);
  const endTime = performance.now();

  console.log(
    `\n⏱️  Search completed in ${(endTime - startTime).toFixed(2)}ms`,
  );
  console.log(`📊 Found ${results.length} results:\n`);

  // 7. Display results
  if (results.length === 0) {
    console.log("No results found. Try a different query.\n");
  } else {
    results.forEach((result, i) => {
      console.log(
        `${
          i + 1
        }. ${result.relativePath}:${result.startLine}-${result.endLine}`,
      );
      console.log(`   Language: ${result.language}`);
      console.log(`   Score: ${result.score.toFixed(4)}`);
      console.log(
        `   Preview: ${
          result.content
            .substring(0, 150)
            .replace(/\n/g, " ")
        }...`,
      );
      console.log();
    });
  }

  console.log("✅ Search example completed successfully!");
}

// Run the example
if (import.meta.main) {
  try {
    await main();
  } catch (error) {
    console.error(
      "\n❌ Error:",
      error instanceof Error ? error.message : error,
    );
    Deno.exit(1);
  }
}
