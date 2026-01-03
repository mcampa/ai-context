# @mcampa/ai-context-cli

A command-line tool for indexing codebases with AI-powered semantic search. This CLI tool reads your configuration and indexes your codebase into a vector database for semantic search capabilities.

## Supported Vector Databases

- **Milvus** - Self-hosted or Zilliz Cloud
- **Qdrant** - Self-hosted or Qdrant Cloud

## Installation

You can run this tool directly without installation using `npx`:

```bash
npx @mcampa/ai-context-cli
```

Or install it globally:

```bash
npm install -g @mcampa/ai-context-cli
```

Or as a dev dependency in your project:

```bash
npm install -D @mcampa/ai-context-cli
```

## Quick Start

1. Create a configuration file in your project root. You can use TypeScript (`.ts`), JavaScript (`.js`), or JSON (`.json`):

**TypeScript/JavaScript projects with Milvus** - `ai-context.config.ts`:

```typescript
import type { ContextConfig } from "@mcampa/ai-context-cli";

const config: ContextConfig = {
  name: "my-project",
  embeddingConfig: {
    apiKey: process.env.OPENAI_API_KEY!,
    model: "text-embedding-3-small",
  },
  vectorDatabaseConfig: {
    address: "localhost:19530",
    // Or for Zilliz Cloud:
    // token: process.env.ZILLIZ_TOKEN,
  },
};

export default config;
```

**TypeScript/JavaScript projects with Qdrant** - `ai-context.config.ts`:

```typescript
import type { ContextConfig } from "@mcampa/ai-context-cli";

const config: ContextConfig = {
  name: "my-project",
  embeddingConfig: {
    apiKey: process.env.OPENAI_API_KEY!,
    model: "text-embedding-3-small",
  },
  vectorDatabaseType: "qdrant",
  qdrantConfig: {
    url: "http://localhost:6333",
    // Or for Qdrant Cloud:
    // url: "https://your-cluster.qdrant.io",
    // apiKey: process.env.QDRANT_API_KEY,
  },
};

export default config;
```

**Non-JS/TS projects (Python, Go, Rust, etc.)** - `ai-context.config.json`:

```json
{
  "name": "my-project",
  "embeddingConfig": {
    "apiKey": "[OPENAI_API_KEY]",
    "model": "text-embedding-3-small"
  },
  "vectorDatabaseConfig": {
    "address": "localhost:19530"
  }
}
```

2. Run the indexer:

```bash
npx @mcampa/ai-context-cli
```

## Configuration

### TypeScript Configuration (`ai-context.config.ts`)

#### With Milvus (default)

```typescript
import type { ContextConfig } from "@mcampa/ai-context-cli";

const config: ContextConfig = {
  // Optional: Name identifier for this context (used in collection naming)
  name: "my-project",

  // Required: OpenAI embedding configuration
  embeddingConfig: {
    apiKey: process.env.OPENAI_API_KEY!,
    model: "text-embedding-3-small",
    // Optional: Custom base URL for OpenAI-compatible APIs
    baseURL: "https://api.openai.com/v1",
  },

  // Optional: Specify vector database type (default: "milvus")
  // vectorDatabaseType: "milvus",

  // Required: Milvus vector database configuration
  vectorDatabaseConfig: {
    // Option 1: Direct address (for self-hosted Milvus)
    address: "localhost:19530",

    // Option 2: Token-based auth (for Zilliz Cloud)
    // token: process.env.ZILLIZ_TOKEN,

    // Optional: Username/password authentication
    // username: "user",
    // password: "password",

    // Optional: Enable SSL
    // ssl: true,
  },

  // Optional: Override default supported file extensions
  // supportedExtensions: [".ts", ".tsx", ".js", ".jsx", ".py"],

  // Optional: Override default ignore patterns
  // ignorePatterns: ["node_modules/**", "dist/**"],

  // Optional: Add additional extensions beyond defaults
  customExtensions: [".vue", ".svelte"],

  // Optional: Add additional ignore patterns beyond defaults
  customIgnorePatterns: ["*.test.ts", "*.spec.ts"],
};

export default config;
```

#### With Qdrant

```typescript
import type { ContextConfig } from "@mcampa/ai-context-cli";

const config: ContextConfig = {
  name: "my-project",

  embeddingConfig: {
    apiKey: process.env.OPENAI_API_KEY!,
    model: "text-embedding-3-small",
  },

  // Use Qdrant as the vector database
  vectorDatabaseType: "qdrant",

  // Required: Qdrant configuration
  qdrantConfig: {
    // Qdrant server URL
    url: "http://localhost:6333",

    // For Qdrant Cloud, include the API key:
    // url: "https://your-cluster.qdrant.io",
    // apiKey: process.env.QDRANT_API_KEY,

    // Optional: Connection timeout in milliseconds (default: 10000)
    // timeout: 30000,
  },
};

export default config;
```

### JavaScript Configuration (`ai-context.config.js`)

```javascript
/** @type {import('@mcampa/ai-context-cli').ContextConfig} */
const config = {
  name: "my-project",
  embeddingConfig: {
    apiKey: process.env.OPENAI_API_KEY,
    model: "text-embedding-3-small",
  },
  vectorDatabaseConfig: {
    address: "localhost:19530",
  },
};

export default config;
```

### JSON Configuration (`ai-context.config.json`)

For non-JavaScript/TypeScript projects (Python, Go, Rust, etc.), you can use a JSON config file:

#### With Milvus

```json
{
  "name": "my-project",
  "embeddingConfig": {
    "apiKey": "[OPENAI_API_KEY]",
    "model": "text-embedding-3-small"
  },
  "vectorDatabaseConfig": {
    "token": "[ZILLIZ_TOKEN]",
    "address": "localhost:19530"
  },
  "supportedExtensions": [".py", ".go", ".rs"],
  "ignorePatterns": ["venv/**", "__pycache__/**", "target/**"],
  "customExtensions": [".proto"],
  "customIgnorePatterns": ["*_test.go"]
}
```

#### With Qdrant

```json
{
  "name": "my-project",
  "embeddingConfig": {
    "apiKey": "[OPENAI_API_KEY]",
    "model": "text-embedding-3-small"
  },
  "vectorDatabaseType": "qdrant",
  "qdrantConfig": {
    "url": "[QDRANT_URL]",
    "apiKey": "[QDRANT_API_KEY]"
  }
}
```

#### Environment Variable Substitution

JSON configs support environment variable substitution using the `[ENV_VAR_NAME]` syntax. Any value containing `[VARIABLE_NAME]` will be replaced with the corresponding environment variable:

```json
{
  "embeddingConfig": {
    "apiKey": "[OPENAI_API_KEY]",
    "baseURL": "https://[API_HOST]:[API_PORT]/v1"
  }
}
```

The CLI will throw an error if a referenced environment variable is not set, ensuring you don't accidentally run with missing configuration

## CLI Options

```
Usage: ai-context-index [options]

Options:
  -V, --version        output the version number
  -c, --config <path>  path to config file (default: ai-context.config.ts/js/json)
  -f, --force          force reindex even if collection already exists
  -h, --help           display help for command
```

### Examples

Index with default config location:

```bash
npx @mcampa/ai-context-cli
```

Specify a custom config file:

```bash
npx @mcampa/ai-context-cli --config ./configs/production.config.ts
```

Force reindex (drop and recreate collection):

```bash
npx @mcampa/ai-context-cli --force
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Index Codebase

on:
  push:
    branches: [main]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "20"

      - name: Index codebase
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ZILLIZ_TOKEN: ${{ secrets.ZILLIZ_TOKEN }}
        run: npx @mcampa/ai-context-cli --force
```

### GitLab CI

```yaml
index-codebase:
  image: node:20
  script:
    - npx @mcampa/ai-context-cli --force
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
    ZILLIZ_TOKEN: $ZILLIZ_TOKEN
  only:
    - main
```

## Environment Variables

You can use environment variables in your configuration file:

| Variable          | Description                            |
| ----------------- | -------------------------------------- |
| `OPENAI_API_KEY`  | Your OpenAI API key                    |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible API base URL  |
| `ZILLIZ_TOKEN`    | Zilliz Cloud API token                 |
| `MILVUS_ADDRESS`  | Milvus server address                  |
| `QDRANT_URL`      | Qdrant server URL                      |
| `QDRANT_API_KEY`  | Qdrant Cloud API key                   |
| `DEBUG`           | Enable debug output (set to any value) |

## Supported File Types

By default, the following file extensions are indexed:

- **TypeScript/JavaScript**: `.ts`, `.tsx`, `.js`, `.jsx`
- **Python**: `.py`
- **Java**: `.java`
- **C/C++**: `.c`, `.cpp`, `.h`, `.hpp`
- **C#**: `.cs`
- **Go**: `.go`
- **Rust**: `.rs`
- **PHP**: `.php`
- **Ruby**: `.rb`
- **Swift**: `.swift`
- **Kotlin**: `.kt`
- **Scala**: `.scala`
- **Objective-C**: `.m`, `.mm`
- **Markdown**: `.md`, `.markdown`
- **Jupyter**: `.ipynb`

Use `customExtensions` to add more file types or `supportedExtensions` to override the defaults.

## Troubleshooting

### "Config file not found"

Make sure you have a config file named `ai-context.config.ts`, `ai-context.config.js`, or `ai-context.config.json` in the directory where you're running the command, or specify a custom path with `--config`.

### "Missing required field: embeddingConfig.apiKey"

Your OpenAI API key is not set. Make sure the environment variable is available:

```bash
export OPENAI_API_KEY="your-api-key"
npx @mcampa/ai-context-cli
```

### "Address is required and could not be resolved from token"

Either provide a Milvus `address` or a Zilliz Cloud `token` in your config.

### Connection errors

- For self-hosted Milvus: Ensure Milvus is running and accessible at the configured address
- For Zilliz Cloud: Verify your token is correct and not expired
- For self-hosted Qdrant: Ensure Qdrant is running and accessible at the configured URL (default port: 6333)
- For Qdrant Cloud: Verify your API key is correct and the cluster URL is valid

### "Missing required field: qdrantConfig.url"

When using Qdrant, the `url` field is required. Make sure to specify the complete URL:

```typescript
qdrantConfig: {
  url: "http://localhost:6333",
}
```

## License

MIT
