# Zuul Pipeline Analyzer with DeepWiki Integration


## Installation

### Requirements

- Python 3.8+
- PyYAML

```bash
pip install pyyaml
```

### Optional: AI Enhancement

For AI-enhanced documentation, also install:

```bash
pip install httpx
```

And set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Quick Start

### List Available Pipelines

```bash
./zuul_deepwiki_analyzer.py /path/to/repo --list
```

### Generate Markdown Documentation

```bash
./zuul_deepwiki_analyzer.py /path/to/repo pipeline-name -o report.md
```

### Generate DeepWiki-Compatible JSON

```bash
./zuul_deepwiki_analyzer.py /path/to/repo pipeline-name --format deepwiki -o wiki.json
```

### With AI Enhancement

```bash
export ANTHROPIC_API_KEY=your_key
./zuul_deepwiki_analyzer.py /path/to/repo pipeline-name --ai-enhance -o report.md
```

## Usage Examples

### Example 1: Basic Analysis

```bash
# List all pipelines in the repository
./zuul_deepwiki_analyzer.py ./ci-framework-jobs --list
```

### Example 2: Generate Full Report

```bash
./zuul_deepwiki_analyzer.py ./ci-framework-jobs \
    openstack-uni01alpha-adoption-jobs-periodic-integration-rhoso18.0-rhel9 \
    -o adoption-pipeline-docs.md
```

### Example 3: DeepWiki Integration

```bash
# Generate JSON for DeepWiki import
./zuul_deepwiki_analyzer.py ./ci-framework-jobs \
    openstack-uni01alpha-adoption-jobs-periodic-integration-rhoso18.0-rhel9 \
    --format deepwiki \
    -o ~/.adalflow/wikicache/adoption-pipeline.json
```



## Command Line Options

```
usage: zuul_deepwiki_analyzer.py [-h] [-l] [-o OUTPUT] [-f {markdown,deepwiki}]
                                  [--ai-enhance] [--deepwiki-url DEEPWIKI_URL]
                                  [--images] [--download-images]
                                  repo_path [pipeline_name]

Analyze Zuul CI/CD pipelines and generate DeepWiki-compatible documentation

positional arguments:
  repo_path             Path to the repository containing Zuul configurations
  pipeline_name         Name of the pipeline to analyze

options:
  -h, --help            show this help message and exit
  -l, --list            List all available pipelines
  -o OUTPUT, --output OUTPUT
                        Output file path (default: stdout)
  -f {markdown,deepwiki}, --format {markdown,deepwiki}
                        Output format: 'markdown' for readable docs, 'deepwiki' for JSON import
  --ai-enhance          Use AI to enhance documentation (requires ANTHROPIC_API_KEY env var)
  --deepwiki-url DEEPWIKI_URL
                        URL of running DeepWiki instance for AI features
  --images              Generate image URLs instead of Mermaid code blocks (uses mermaid.ink)
  --download-images     Download images as local SVG files (requires --images and httpx)
```

## How It Works

1. **Discovery Phase**
   - Scans repository for Zuul YAML files
   - Parses job definitions, project-templates, and projects
   - Builds a registry of all jobs and pipelines

2. **Analysis Phase**
   - Resolves job inheritance chains
   - Collects all playbooks (pre-run, run, post-run) for each job
   - Parses Ansible playbooks to extract tasks, roles, and handlers

3. **Generation Phase**
   - Calculates parallel execution stages
   - Generates Mermaid diagrams
   - Optionally calls AI for enhanced summaries
   - Produces output in requested format
