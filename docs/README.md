# Documentation

This directory contains the source files for the Claude Code Proxy API documentation, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Quick Start

### Prerequisites

- Python 3.11+
- uv (preferred) or pip

### Setup and Serve Locally

```bash
# Install documentation dependencies
make docs-install

# Serve documentation with live reload
make docs-serve
```

The documentation will be available at `http://127.0.0.1:8080`.

### Build for Production

```bash
# Build static documentation
make docs-build

# Clean build artifacts
make docs-clean
```

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/            # Installation and setup
│   ├── quickstart.md          # Quick start guide
│   ├── installation.md        # Detailed installation
│   └── configuration.md       # Configuration reference
├── user-guide/                # User documentation
│   ├── api-usage.md           # API usage examples
│   ├── authentication.md     # Authentication guide
│   ├── streaming.md           # Streaming documentation
│   └── error-handling.md      # Error handling guide
├── api-reference/              # API documentation
│   ├── overview.md            # API overview
│   ├── anthropic.md           # Anthropic endpoints
│   ├── openai.md              # OpenAI endpoints
│   ├── models.md              # Available models
│   └── health.md              # Health endpoints
├── developer-guide/            # Developer documentation
│   ├── architecture.md        # System architecture
│   ├── development.md         # Development setup
│   ├── testing.md             # Testing guide
│   └── contributing.md        # Contribution guidelines
├── deployment/                 # Deployment guides
│   ├── docker.md              # Docker deployment
│   ├── production.md          # Production deployment
│   ├── kubernetes.md          # Kubernetes deployment
│   └── monitoring.md          # Monitoring setup
├── examples/                   # Code examples
│   ├── python-client.md       # Python examples
│   ├── curl.md                # curl examples
│   ├── javascript.md          # JavaScript examples
│   └── openai-sdk.md          # OpenAI SDK examples
├── reference/                  # Auto-generated API docs
│   └── (generated from source code)
└── assets/                     # Static assets
    ├── extra.css              # Custom CSS
    ├── extra.js               # Custom JavaScript
    └── images/                # Images and icons
```

## Features

### Core Features
- **Material Design**: Modern, responsive theme with dark/light mode
- **Auto-generated API docs**: Documentation generated from Python source code
- **Search**: Full-text search across all documentation
- **Code highlighting**: Syntax highlighting for multiple languages
- **Copy buttons**: One-click code copying
- **Mobile responsive**: Optimized for mobile and desktop

### Advanced Features
- **Live reload**: Automatic page refresh during development
- **Mermaid diagrams**: Support for flowcharts and diagrams
- **OpenAPI/Swagger**: Interactive API documentation
- **GitHub integration**: Edit buttons linking to source files
- **Version management**: Support for multiple documentation versions

### Plugins Enabled
- **mkdocstrings**: Auto-generated Python API documentation
- **search**: Enhanced search functionality
- **section-index**: Automatic section indices
- **swagger-ui-tag**: Interactive API documentation
- **mermaid2**: Diagram support
- **glightbox**: Image lightbox
- **minify**: CSS/JS minification for production

## Configuration

The documentation is configured in `mkdocs.yml` in the project root. Key configuration sections:

- **Theme**: Material theme with custom colors and features
- **Navigation**: Structured site navigation
- **Plugins**: Documentation generation and enhancement plugins
- **Markdown extensions**: Enhanced Markdown syntax support

## Auto-Generated Documentation

API documentation is automatically generated from Python source code using mkdocstrings. The generation script is located at:

```
docs/gen_ref_pages.py
```

This script scans the `ccproxy/` package and generates documentation pages for all modules, classes, and functions.

## Custom Styling

Custom CSS and JavaScript are included for enhanced functionality:

- `docs/assets/extra.css`: Custom styling for API endpoints, status codes, and layout
- `docs/assets/extra.js`: Interactive features like copy buttons and navigation enhancements

## GitHub Actions

Automated documentation building and deployment is configured in:

```
.github/workflows/docs.yml
```

This workflow:
- Builds documentation on every push to main/dev branches
- Deploys to GitHub Pages automatically
- Runs link checking and validation
- Supports manual deployment triggers

## Commands Reference

### Make Commands

```bash
make docs-install   # Install documentation dependencies
make docs-serve     # Start development server
make docs-build     # Build static documentation
make docs-clean     # Clean build artifacts
make docs-deploy    # Deploy documentation (GitHub Pages)
```

### Direct MkDocs Commands

```bash
# Install dependencies
uv sync --group docs

# Development server
mkdocs serve --dev-addr 127.0.0.1:8080

# Build documentation
mkdocs build --clean

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Contributing to Documentation

### Adding New Pages

1. Create a new Markdown file in the appropriate directory
2. Add the page to the navigation in `mkdocs.yml`
3. Follow the existing styling conventions
4. Test locally with `make docs-serve`

### Updating API Documentation

API documentation is automatically generated from source code. To update:

1. Add/update docstrings in Python source files
2. Use Google-style docstrings for best formatting
3. Rebuild documentation to see changes

### Style Guidelines

- Use clear, concise language
- Include practical examples
- Follow the existing structure and formatting
- Use code blocks with appropriate language tags
- Include navigation links between related pages

## Troubleshooting

### Common Issues

1. **Build errors**: Check that all dependencies are installed with `make docs-install`
2. **Missing pages**: Ensure new pages are added to the navigation in `mkdocs.yml`
3. **Broken links**: Use relative paths and check link targets exist
4. **Styling issues**: Check custom CSS in `docs/assets/extra.css`

### Development Tips

- Use `mkdocs serve` for live reload during development
- Check the browser console for JavaScript errors
- Validate Markdown syntax before committing
- Test on both desktop and mobile viewports

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)
- [Markdown Guide](https://www.markdownguide.org/)

---

For questions about the documentation system, please open an issue in the GitHub repository.
