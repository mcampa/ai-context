#!/usr/bin/env node

/**
 * Release script for @mcampa/ai-context packages
 *
 * Usage: pnpm release v0.0.2
 *
 * This script:
 * 1. Validates the version is a valid semver
 * 2. Validates the version is higher than the current version
 * 3. Validates all three packages are on the same version
 * 4. Updates all package.json files
 * 5. Commits the changes
 * 6. Creates a git tag
 * 7. Pushes the commit and tag
 */

import { readFileSync, writeFileSync } from "fs";
import { execSync } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, "..");

// Package paths relative to root
const PACKAGES = [
  "packages/core/package.json",
  "packages/mcp/package.json",
  "packages/cli/package.json",
];

/**
 * Parse semver version string
 * @param {string} version
 * @returns {{ major: number, minor: number, patch: number, prerelease?: string } | null}
 */
function parseSemver(version) {
  // Remove leading 'v' if present
  const cleanVersion = version.startsWith("v") ? version.slice(1) : version;

  // Match semver pattern: major.minor.patch[-prerelease]
  const match = cleanVersion.match(
    /^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?$/,
  );

  if (!match) {
    return null;
  }

  return {
    major: parseInt(match[1], 10),
    minor: parseInt(match[2], 10),
    patch: parseInt(match[3], 10),
    prerelease: match[4] || undefined,
  };
}

/**
 * Compare two semver versions
 * @param {string} v1
 * @param {string} v2
 * @returns {number} -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
 */
function compareSemver(v1, v2) {
  const parsed1 = parseSemver(v1);
  const parsed2 = parseSemver(v2);

  if (!parsed1 || !parsed2) {
    throw new Error(`Invalid semver: ${!parsed1 ? v1 : v2}`);
  }

  // Compare major.minor.patch
  if (parsed1.major !== parsed2.major) {
    return parsed1.major > parsed2.major ? 1 : -1;
  }
  if (parsed1.minor !== parsed2.minor) {
    return parsed1.minor > parsed2.minor ? 1 : -1;
  }
  if (parsed1.patch !== parsed2.patch) {
    return parsed1.patch > parsed2.patch ? 1 : -1;
  }

  // If one has prerelease and other doesn't, the one without is higher
  if (parsed1.prerelease && !parsed2.prerelease) {
    return -1;
  }
  if (!parsed1.prerelease && parsed2.prerelease) {
    return 1;
  }

  // Compare prerelease strings lexically
  if (parsed1.prerelease && parsed2.prerelease) {
    return parsed1.prerelease.localeCompare(parsed2.prerelease);
  }

  return 0;
}

/**
 * Read package.json and return parsed content
 * @param {string} relativePath
 * @returns {{ path: string, content: object }}
 */
function readPackageJson(relativePath) {
  const fullPath = join(rootDir, relativePath);
  const content = JSON.parse(readFileSync(fullPath, "utf-8"));
  return { path: fullPath, relativePath, content };
}

/**
 * Write package.json
 * @param {string} fullPath
 * @param {object} content
 */
function writePackageJson(fullPath, content) {
  writeFileSync(fullPath, JSON.stringify(content, null, 2) + "\n");
}

/**
 * Execute a command and return output
 * @param {string} command
 * @returns {string}
 */
function exec(command) {
  return execSync(command, { encoding: "utf-8", cwd: rootDir }).trim();
}

/**
 * Check if git working directory is clean
 * @returns {boolean}
 */
function isGitClean() {
  try {
    const status = exec("git status --porcelain");
    return status === "";
  } catch {
    return false;
  }
}

/**
 * Main release function
 */
function release() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error("❌ Error: Version argument required");
    console.error("Usage: pnpm release v0.0.2");
    process.exit(1);
  }

  const versionArg = args[0];

  // Ensure version starts with 'v'
  const tagVersion = versionArg.startsWith("v") ? versionArg : `v${versionArg}`;
  const version = tagVersion.slice(1); // Remove 'v' for package.json

  console.log("");
  console.log("🚀 Starting release process...");
  console.log("");

  // Step 1: Validate semver
  console.log(`📋 Validating version: ${tagVersion}`);
  const parsed = parseSemver(version);
  if (!parsed) {
    console.error(`❌ Error: Invalid semver version: ${version}`);
    console.error(
      "   Version must be in format: major.minor.patch[-prerelease]",
    );
    console.error("   Examples: 1.0.0, 1.2.3, 2.0.0-beta.1");
    process.exit(1);
  }
  console.log(`   ✅ Valid semver: ${version}`);

  // Step 2: Read all package.json files
  console.log("");
  console.log("📦 Reading package versions...");
  const packages = PACKAGES.map(readPackageJson);

  // Step 3: Validate all packages are on the same version
  const versions = packages.map((p) => p.content.version);
  const uniqueVersions = [...new Set(versions)];

  if (uniqueVersions.length > 1) {
    console.error("❌ Error: Packages are not on the same version:");
    packages.forEach((p) => {
      console.error(`   ${p.relativePath}: ${p.content.version}`);
    });
    console.error("");
    console.error(
      "   Please ensure all packages have the same version before releasing.",
    );
    process.exit(1);
  }

  const currentVersion = uniqueVersions[0];
  console.log(`   Current version: ${currentVersion}`);
  console.log(`   New version: ${version}`);

  // Step 4: Validate new version is higher or equal
  const comparison = compareSemver(version, currentVersion);
  if (comparison < 0) {
    console.error("");
    console.error(
      `❌ Error: New version (${version}) must be higher than or equal to current version (${currentVersion})`,
    );
    process.exit(1);
  }
  if (comparison === 0) {
    console.log(
      `   ⚠️  Version ${version} is the same as current version (re-release)`,
    );
  } else {
    console.log(`   ✅ Version ${version} is higher than ${currentVersion}`);
  }

  // Step 5: Check git status
  console.log("");
  console.log("🔍 Checking git status...");
  if (!isGitClean()) {
    console.error(
      "❌ Error: Git working directory is not clean. Please commit or stash changes first.",
    );
    process.exit(1);
  }
  console.log("   ✅ Git working directory is clean");

  // Step 6: Check if tag already exists
  let tagExists = false;
  try {
    execSync(`git rev-parse ${tagVersion}`, {
      encoding: "utf-8",
      cwd: rootDir,
      stdio: "pipe", // Suppress stderr output
    });
    tagExists = true;
  } catch {
    // Tag doesn't exist, which is what we want
  }

  if (tagExists) {
    console.error(`❌ Error: Tag ${tagVersion} already exists`);
    process.exit(1);
  }

  // Step 7: Update package.json files (skip if same version)
  const isReRelease = comparison === 0;

  if (isReRelease) {
    console.log("");
    console.log("📝 Skipping package.json updates (same version)...");
  } else {
    console.log("");
    console.log("📝 Updating package.json files...");
    packages.forEach((p) => {
      p.content.version = version;
      writePackageJson(p.path, p.content);
      console.log(`   ✅ Updated ${p.relativePath}`);
    });

    // Step 8: Commit changes
    console.log("");
    console.log("💾 Committing changes...");
    try {
      exec("git add packages/*/package.json");
      exec(`git commit -m "chore: release ${tagVersion}"`);
      console.log(`   ✅ Committed: chore: release ${tagVersion}`);
    } catch (error) {
      console.error("❌ Error: Failed to commit changes");
      console.error(error.message);
      process.exit(1);
    }
  }

  // Step 9: Create tag
  console.log("");
  console.log("🏷️  Creating git tag...");
  try {
    exec(`git tag ${tagVersion}`);
    console.log(`   ✅ Created tag: ${tagVersion}`);
  } catch (error) {
    console.error(`❌ Error: Failed to create tag ${tagVersion}`);
    console.error(error.message);
    process.exit(1);
  }

  // Step 10: Push commit and tag
  console.log("");
  console.log("🚀 Pushing to remote...");
  try {
    if (!isReRelease) {
      exec("git push");
    }
    exec(`git push origin ${tagVersion}`);
    console.log(
      isReRelease ? "   ✅ Pushed tag" : "   ✅ Pushed commit and tag",
    );
  } catch (error) {
    console.error("❌ Error: Failed to push to remote");
    console.error(error.message);
    console.error("");
    console.error("You can manually push with:");
    if (!isReRelease) {
      console.error("   git push");
    }
    console.error(`   git push origin ${tagVersion}`);
    process.exit(1);
  }

  // Done!
  console.log("");
  console.log("═".repeat(50));
  console.log(`🎉 Successfully released ${tagVersion}!`);
  console.log("═".repeat(50));
  console.log("");
  console.log("The GitHub Actions workflow will now:");
  console.log("  1. Build all packages");
  console.log("  2. Publish to npm:");
  console.log("     - @mcampa/ai-context-core");
  console.log("     - @mcampa/ai-context-mcp");
  console.log("     - @mcampa/ai-context-cli");
  console.log("");
  console.log("Check the workflow status at:");
  console.log("  https://github.com/mcampa/ai-context/actions");
  console.log("");
}

release();
