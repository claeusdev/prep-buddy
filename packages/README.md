# PrepBuddy Shared Libraries

This directory contains reusable libraries extracted from PrepBuddy that can be used across multiple projects.

## Packages

### [@prepbuddy/gemini-service](./gemini-service)
Google Gemini AI service wrapper for coding education and interview preparation.

**Features:**
- Solution analysis and grading
- Official solution generation
- Problem explanations
- Coding pattern identification
- Interactive tutoring (coding, system design, learning)

### [@prepbuddy/storage](./storage)
Progress storage service with localStorage.

**Features:**
- Local storage for progress tracking
- Question progress tracking
- Offline support
- Code and chat history persistence

## Development

### Building All Packages

```bash
# Build all packages once
npm run build:packages

# Watch mode for all packages
npm run dev:packages
```

### Building Individual Packages

```bash
# Navigate to specific package
cd packages/gemini-service

# Build
npm run build

# Watch mode
npm run dev
```

### Installing Dependencies

```bash
# From root directory - installs all package dependencies
npm install
```

## Using These Libraries in Other Projects

### Option 1: Local Development (npm link)

```bash
# In the package directory
cd packages/gemini-service
npm link

# In your other project
npm link @prepbuddy/gemini-service
```

### Option 2: Publish to npm

```bash
cd packages/gemini-service
npm publish --access public
```

### Option 3: Git-based Installation

```json
{
  "dependencies": {
    "@prepbuddy/gemini-service": "github:yourusername/PrepBuddy#main:packages/gemini-service"
  }
}
```

### Option 4: Local Path (Monorepo)

```json
{
  "dependencies": {
    "@prepbuddy/gemini-service": "file:../PrepBuddy/packages/gemini-service"
  }
}
```

## Package Structure

Each package follows this structure:

```
packages/[package-name]/
├── src/
│   └── index.ts          # Main entry point
├── dist/                 # Built files (generated)
│   ├── index.js         # CommonJS bundle
│   ├── index.mjs        # ESM bundle
│   └── index.d.ts       # TypeScript definitions
├── package.json         # Package configuration
├── tsconfig.json        # TypeScript configuration
└── README.md           # Package documentation
```

## License

MIT
