#!/usr/bin/env python3
"""
Zuul Pipeline Analyzer with DeepWiki Integration

Analyzes Zuul CI/CD repository structures and generates documentation
compatible with DeepWiki's wiki format.

Features:
- Parse Zuul job definitions and pipelines
- Resolve job inheritance chains
- Extract and summarize Ansible playbooks
- Generate Mermaid diagrams
- Output DeepWiki-compatible JSON for import
- Optional AI-enhanced documentation via Anthropic API or DeepWiki API

Usage:
    # Basic analysis (Markdown output)
    ./zuul_deepwiki_analyzer.py <repo_path> <pipeline_name>
    
    # DeepWiki JSON output
    ./zuul_deepwiki_analyzer.py <repo_path> <pipeline_name> --format deepwiki
    
    # With AI enhancement (requires ANTHROPIC_API_KEY)
    ./zuul_deepwiki_analyzer.py <repo_path> <pipeline_name> --ai-enhance
    
    # Connect to running DeepWiki instance
    ./zuul_deepwiki_analyzer.py <repo_path> <pipeline_name> --deepwiki-url http://localhost:8001

Example:
    ./zuul_deepwiki_analyzer.py ./my-repo openstack-uni01alpha-adoption-jobs-periodic-integration-rhoso18.0-rhel9
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import re
import zlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
from urllib.parse import quote
import yaml

# Optional imports for AI enhancement and image generation
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class MermaidImageGenerator:
    """Generates images from Mermaid diagrams using mermaid.ink service."""
    
    def __init__(self, output_dir: Path, download_images: bool = False):
        self.output_dir = output_dir
        self.download_images = download_images
        self.images_dir = output_dir / "images"
        if download_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self.image_counter = 0
    
    def _encode_mermaid(self, mermaid_code: str) -> str:
        """Encode mermaid code for mermaid.ink URL."""
        json_str = json.dumps({"code": mermaid_code, "mermaid": {"theme": "default"}})
        compressed = zlib.compress(json_str.encode('utf-8'), level=9)
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
        return encoded
    
    def _encode_mermaid_simple(self, mermaid_code: str) -> str:
        """Simple base64 encoding for mermaid.ink."""
        return base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('ascii')
    
    def get_image_url(self, mermaid_code: str, format: str = "svg") -> str:
        """Get mermaid.ink URL for the diagram."""
        encoded = self._encode_mermaid_simple(mermaid_code)
        return f"https://mermaid.ink/img/{encoded}?type={format}"
    
    def generate_image(self, mermaid_code: str, name_hint: str = "") -> tuple[str, str]:
        """
        Generate an image from Mermaid code.
        
        Returns:
            tuple: (image_path_or_url, markdown_embed_syntax)
        """
        self.image_counter += 1
        
        # Create a clean filename
        if name_hint:
            clean_name = re.sub(r'[^a-z0-9]+', '_', name_hint.lower())[:40]
            filename = f"diagram_{self.image_counter:02d}_{clean_name}"
        else:
            filename = f"diagram_{self.image_counter:02d}"
        
        # Generate URL
        url = self.get_image_url(mermaid_code, "svg")
        
        if self.download_images and HTTPX_AVAILABLE:
            # Try to download the image
            try:
                with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    
                    # Save as SVG
                    svg_path = self.images_dir / f"{filename}.svg"
                    svg_path.write_bytes(response.content)
                    
                    # Return relative path for markdown
                    rel_path = f"images/{filename}.svg"
                    return rel_path, f"![{name_hint or 'Diagram'}]({rel_path})"
            except Exception as e:
                print(f"Warning: Could not download image: {e}", file=sys.stderr)
                # Fall back to URL
                return url, f"![{name_hint or 'Diagram'}]({url})"
        else:
            # Return URL-based embed
            return url, f"![{name_hint or 'Diagram'}]({url})"
    
    def mermaid_to_markdown(self, mermaid_code: str, name_hint: str = "") -> str:
        """
        Convert Mermaid code to markdown image syntax.
        
        Args:
            mermaid_code: The Mermaid diagram code
            name_hint: A descriptive name for the diagram
            
        Returns:
            Markdown image syntax with mermaid.ink URL
        """
        _, markdown = self.generate_image(mermaid_code, name_hint)
        return markdown


@dataclass
class PlaybookSummary:
    """Summary of an Ansible playbook."""
    path: str
    name: str = ""
    hosts: str = ""
    tasks: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    handlers: list = field(default_factory=list)
    vars_defined: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    includes: list = field(default_factory=list)
    description: str = ""
    exists: bool = True
    error: str = ""
    ai_summary: str = ""  # AI-generated summary


@dataclass
class ZuulJob:
    """Representation of a Zuul job definition."""
    name: str
    parent: Optional[str] = None
    abstract: bool = False
    description: str = ""
    pre_run: list = field(default_factory=list)
    run: list = field(default_factory=list)
    post_run: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    required_projects: list = field(default_factory=list)
    vars: dict = field(default_factory=dict)
    nodeset: str = ""
    timeout: int = 0
    source_file: str = ""
    ai_summary: str = ""  # AI-generated summary


@dataclass
class PipelineJob:
    """A job within a pipeline with its dependencies."""
    name: str
    dependencies: list = field(default_factory=list)


@dataclass
class Pipeline:
    """A pipeline (project-template) definition."""
    name: str
    jobs: list = field(default_factory=list)


@dataclass
class WikiPage:
    """A page in the DeepWiki format."""
    id: str
    title: str
    content: str
    importance: int = 1
    file_hashes: list = field(default_factory=list)
    parent: Optional[str] = None


@dataclass
class WikiStructure:
    """DeepWiki-compatible wiki structure."""
    pages: list = field(default_factory=list)
    repo_url: str = ""
    generated_at: str = ""
    analyzer_version: str = "1.0.0"


class AIEnhancer:
    """Handles AI-enhanced documentation generation."""
    
    def __init__(self, anthropic_api_key: str = None, deepwiki_url: str = None):
        self.anthropic_api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.deepwiki_url = deepwiki_url
        self.enabled = bool(self.anthropic_api_key or self.deepwiki_url)
        
        if not HTTPX_AVAILABLE and self.enabled:
            print("Warning: httpx not installed. AI enhancement disabled.", file=sys.stderr)
            print("Install with: pip install httpx", file=sys.stderr)
            self.enabled = False
    
    def summarize_playbook(self, playbook_content: str, playbook_path: str) -> str:
        """Generate an AI summary for a playbook."""
        if not self.enabled:
            return ""
        
        prompt = f"""Analyze this Ansible playbook and provide a concise 2-3 sentence summary of what it does.
Focus on the main purpose, key tasks, and any important configurations.

Playbook path: {playbook_path}

```yaml
{playbook_content[:4000]}  # Truncate for token limits
```

Provide only the summary, no additional formatting."""

        return self._call_ai(prompt)
    
    def summarize_job(self, job: ZuulJob, playbook_summaries: list) -> str:
        """Generate an AI summary for a Zuul job."""
        if not self.enabled:
            return ""
        
        playbooks_info = "\n".join([
            f"- {ps.path}: {ps.description or 'No description'}"
            for ps in playbook_summaries
        ])
        
        prompt = f"""Analyze this Zuul CI job and provide a concise 2-3 sentence summary of its purpose and what it accomplishes.

Job Name: {job.name}
Parent: {job.parent or 'None'}
Description: {job.description or 'None provided'}
Timeout: {job.timeout}s
Nodeset: {job.nodeset}

Pre-run playbooks: {', '.join(job.pre_run) or 'None'}
Run playbooks: {', '.join(job.run) or 'None'}  
Post-run playbooks: {', '.join(job.post_run) or 'None'}

Playbook summaries:
{playbooks_info}

Key variables: {list(job.vars.keys())[:10]}

Provide only the summary, no additional formatting."""

        return self._call_ai(prompt)
    
    def summarize_pipeline(self, pipeline: Pipeline, job_summaries: dict) -> str:
        """Generate an AI summary for a pipeline."""
        if not self.enabled:
            return ""
        
        jobs_info = "\n".join([
            f"- {job.name}: {job_summaries.get(job.name, 'No summary')}"
            for job in pipeline.jobs
        ])
        
        prompt = f"""Analyze this Zuul CI pipeline and provide a comprehensive summary of its purpose and workflow.

Pipeline Name: {pipeline.name}
Number of Jobs: {len(pipeline.jobs)}

Jobs and their purposes:
{jobs_info}

Provide a 3-4 sentence summary explaining:
1. The overall purpose of this pipeline
2. The main workflow stages
3. What the pipeline accomplishes when complete

Provide only the summary, no additional formatting."""

        return self._call_ai(prompt)
    
    def _call_ai(self, prompt: str) -> str:
        """Call the AI API (Anthropic or DeepWiki)."""
        if not self.enabled or not HTTPX_AVAILABLE:
            return ""
        
        try:
            if self.anthropic_api_key:
                return self._call_anthropic(prompt)
            elif self.deepwiki_url:
                return self._call_deepwiki(prompt)
        except Exception as e:
            print(f"Warning: AI call failed: {e}", file=sys.stderr)
            return ""
        
        return ""
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call the Anthropic API directly."""
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
    
    def _call_deepwiki(self, prompt: str) -> str:
        """Call a running DeepWiki instance."""
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.deepwiki_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            response.raise_for_status()
            # Collect streamed response
            return response.text


class ZuulDeepWikiAnalyzer:
    """Analyzes Zuul repositories and generates DeepWiki-compatible documentation."""
    
    def __init__(self, repo_path: str, ai_enhancer: AIEnhancer = None, 
                 output_dir: str = None, generate_images: bool = False,
                 download_images: bool = False):
        self.repo_path = Path(repo_path).resolve()
        self.jobs: dict[str, ZuulJob] = {}
        self.pipelines: dict[str, Pipeline] = {}
        self.playbook_cache: dict[str, PlaybookSummary] = {}
        self.ai = ai_enhancer or AIEnhancer()
        self.generate_images = generate_images
        
        # Setup image generator
        if output_dir:
            self.output_dir = Path(output_dir).parent if output_dir.endswith(('.md', '.json')) else Path(output_dir)
        else:
            self.output_dir = Path.cwd()
        
        self.image_generator = MermaidImageGenerator(self.output_dir, download_images) if generate_images else None
        
    def find_zuul_files(self) -> list[Path]:
        """Find all Zuul YAML configuration files."""
        zuul_files = []
        patterns = [
            ".zuul.yaml",
            "zuul.yaml", 
            "zuul.d/*.yaml",
            "zuul.d/**/*.yaml",
            ".zuul.d/*.yaml",
            ".zuul.d/**/*.yaml",
        ]
        
        for pattern in patterns:
            zuul_files.extend(self.repo_path.glob(pattern))
        
        return sorted(set(zuul_files))
    
    def parse_zuul_files(self):
        """Parse all Zuul configuration files."""
        zuul_files = self.find_zuul_files()
        
        for zuul_file in zuul_files:
            try:
                with open(zuul_file, 'r') as f:
                    content = yaml.safe_load(f)
                    
                if not content:
                    continue
                    
                if not isinstance(content, list):
                    content = [content]
                
                for item in content:
                    if not isinstance(item, dict):
                        continue
                        
                    if 'job' in item:
                        self._parse_job(item['job'], str(zuul_file))
                    elif 'project-template' in item:
                        self._parse_project_template(item['project-template'], str(zuul_file))
                    elif 'project' in item:
                        self._parse_project(item['project'], str(zuul_file))
                        
            except yaml.YAMLError as e:
                print(f"Warning: Error parsing {zuul_file}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error processing {zuul_file}: {e}", file=sys.stderr)
    
    def _parse_job(self, job_data: dict, source_file: str):
        """Parse a single job definition."""
        name = job_data.get('name', '')
        if not name:
            return
            
        job = ZuulJob(
            name=name,
            parent=job_data.get('parent'),
            abstract=job_data.get('abstract', False),
            description=job_data.get('description', '').strip(),
            pre_run=self._ensure_list(job_data.get('pre-run', [])),
            run=self._ensure_list(job_data.get('run', [])),
            post_run=self._ensure_list(job_data.get('post-run', [])),
            roles=job_data.get('roles', []),
            required_projects=job_data.get('required-projects', []),
            vars=job_data.get('vars', {}),
            nodeset=self._extract_nodeset(job_data.get('nodeset', '')),
            timeout=job_data.get('timeout', 0),
            source_file=source_file,
        )
        
        self.jobs[name] = job
    
    def _extract_nodeset(self, nodeset) -> str:
        """Extract nodeset name from various formats."""
        if isinstance(nodeset, str):
            return nodeset
        elif isinstance(nodeset, dict):
            return nodeset.get('name', str(nodeset))
        return str(nodeset) if nodeset else ""
    
    def _ensure_list(self, value) -> list:
        """Ensure a value is a list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
    
    def _parse_project_template(self, template_data: dict, source_file: str):
        """Parse a project-template definition."""
        for key, value in template_data.items():
            if key == 'name':
                continue
            
            if isinstance(value, dict) and 'jobs' in value:
                pipeline = Pipeline(name=key, jobs=[])
                self._parse_pipeline_jobs(value['jobs'], pipeline)
                self.pipelines[key] = pipeline
            elif isinstance(value, dict):
                pipeline = Pipeline(name=key, jobs=[])
                self._parse_pipeline_jobs_from_dict(value, pipeline)
                if pipeline.jobs:
                    self.pipelines[key] = pipeline
    
    def _parse_project(self, project_data: dict, source_file: str):
        """Parse a project definition for pipeline configurations."""
        for key, value in project_data.items():
            if key in ('name', 'default-branch', 'vars'):
                continue
            
            if isinstance(value, dict) and 'jobs' in value:
                pipeline = Pipeline(name=key, jobs=[])
                self._parse_pipeline_jobs(value['jobs'], pipeline)
                self.pipelines[key] = pipeline
    
    def _parse_pipeline_jobs(self, jobs_list: list, pipeline: Pipeline):
        """Parse jobs list in a pipeline."""
        if not jobs_list:
            return
            
        for job_entry in jobs_list:
            if isinstance(job_entry, str):
                pipeline.jobs.append(PipelineJob(name=job_entry))
            elif isinstance(job_entry, dict):
                for job_name, job_config in job_entry.items():
                    deps = []
                    if isinstance(job_config, dict):
                        deps = job_config.get('dependencies', [])
                        if isinstance(deps, str):
                            deps = [deps]
                    pipeline.jobs.append(PipelineJob(name=job_name, dependencies=deps))
    
    def _parse_pipeline_jobs_from_dict(self, data: dict, pipeline: Pipeline):
        """Parse jobs from a dict structure."""
        if 'jobs' in data:
            self._parse_pipeline_jobs(data['jobs'], pipeline)
    
    def resolve_job_inheritance(self, job_name: str) -> list[ZuulJob]:
        """Resolve the full inheritance chain for a job."""
        chain = []
        current_name = job_name
        visited = set()
        
        while current_name and current_name not in visited:
            visited.add(current_name)
            job = self.jobs.get(current_name)
            if job:
                chain.append(job)
                current_name = job.parent
            else:
                chain.append(ZuulJob(name=current_name, description="External job (not defined in this repository)"))
                break
        
        return chain
    
    def get_all_playbooks_for_job(self, job_name: str) -> dict[str, list[str]]:
        """Get all playbooks (pre-run, run, post-run) for a job including inherited ones."""
        chain = self.resolve_job_inheritance(job_name)
        
        result = {
            'pre_run': [],
            'run': [],
            'post_run': [],
        }
        
        for job in reversed(chain):
            if job.pre_run:
                for pb in job.pre_run:
                    if pb not in result['pre_run']:
                        result['pre_run'].append(pb)
            
            if job.run:
                result['run'] = job.run.copy()
            
            if job.post_run:
                for pb in job.post_run:
                    if pb not in result['post_run']:
                        result['post_run'].append(pb)
        
        return result
    
    def parse_playbook(self, playbook_path: str) -> PlaybookSummary:
        """Parse an Ansible playbook and extract summary information."""
        if playbook_path in self.playbook_cache:
            return self.playbook_cache[playbook_path]
        
        full_path = self._find_playbook(playbook_path)
        
        summary = PlaybookSummary(path=playbook_path)
        
        if not full_path or not full_path.exists():
            summary.exists = False
            summary.error = f"Playbook not found: {playbook_path}"
            self.playbook_cache[playbook_path] = summary
            return summary
        
        try:
            with open(full_path, 'r') as f:
                raw_content = f.read()
                content = yaml.safe_load(raw_content)
            
            if not content:
                summary.description = "Empty playbook"
                self.playbook_cache[playbook_path] = summary
                return summary
            
            if isinstance(content, list):
                for play in content:
                    if isinstance(play, dict):
                        self._parse_play(play, summary)
            elif isinstance(content, dict):
                self._parse_play(content, summary)
            
            # AI enhancement
            if self.ai.enabled:
                summary.ai_summary = self.ai.summarize_playbook(raw_content, playbook_path)
                
        except yaml.YAMLError as e:
            summary.error = f"YAML parse error: {e}"
        except Exception as e:
            summary.error = f"Error: {e}"
        
        self.playbook_cache[playbook_path] = summary
        return summary
    
    def _find_playbook(self, playbook_path: str) -> Optional[Path]:
        """Find a playbook file in the repository."""
        direct = self.repo_path / playbook_path
        if direct.exists():
            return direct
        
        for prefix in ['', 'playbooks/', 'ansible/', 'zuul.d/playbooks/']:
            path = self.repo_path / prefix / playbook_path
            if path.exists():
                return path
        
        name = Path(playbook_path).name
        for found in self.repo_path.rglob(name):
            if found.is_file():
                return found
        
        return None
    
    def _parse_play(self, play: dict, summary: PlaybookSummary):
        """Parse a single play from a playbook."""
        if 'name' in play:
            if not summary.name:
                summary.name = play['name']
            else:
                summary.name += f"; {play['name']}"
        
        if 'hosts' in play:
            summary.hosts = play['hosts']
        
        if summary.name and not summary.description:
            summary.description = summary.name
        
        if 'tasks' in play:
            for task in play['tasks']:
                if isinstance(task, dict):
                    task_info = self._extract_task_info(task)
                    if task_info:
                        summary.tasks.append(task_info)
        
        if 'roles' in play:
            for role in play['roles']:
                if isinstance(role, str):
                    summary.roles.append(role)
                elif isinstance(role, dict):
                    role_name = role.get('role') or role.get('name', str(role))
                    summary.roles.append(role_name)
        
        if 'handlers' in play:
            for handler in play['handlers']:
                if isinstance(handler, dict):
                    handler_name = handler.get('name', 'unnamed')
                    summary.handlers.append(handler_name)
        
        if 'import_playbook' in play:
            summary.imports.append(play['import_playbook'])
        if 'include' in play:
            summary.includes.append(play['include'])
        
        if 'vars' in play:
            for var_name in play['vars'].keys():
                summary.vars_defined.append(var_name)
    
    def _extract_task_info(self, task: dict) -> Optional[str]:
        """Extract task information."""
        name = task.get('name', '')
        
        modules = [
            'ansible.builtin.command', 'command',
            'ansible.builtin.shell', 'shell',
            'ansible.builtin.copy', 'copy',
            'ansible.builtin.file', 'file',
            'ansible.builtin.template', 'template',
            'ansible.builtin.package', 'package',
            'ansible.builtin.service', 'service',
            'ansible.builtin.debug', 'debug',
            'ansible.builtin.fail', 'fail',
            'ansible.builtin.set_fact', 'set_fact',
            'ansible.builtin.include_tasks', 'include_tasks',
            'ansible.builtin.import_tasks', 'import_tasks',
            'ansible.builtin.include_role', 'include_role',
            'ansible.builtin.import_role', 'import_role',
            'ansible.builtin.slurp', 'slurp',
            'community.general.make', 'make',
        ]
        
        module_used = None
        for module in modules:
            if module in task:
                module_used = module.split('.')[-1]
                break
        
        if name:
            if module_used:
                return f"{name} [{module_used}]"
            return name
        elif module_used:
            return f"[{module_used}]"
        
        return None
    
    def _calculate_execution_order(self, pipeline: Pipeline) -> list[list[str]]:
        """Calculate the parallel execution stages for jobs."""
        deps = {job.name: set(job.dependencies) for job in pipeline.jobs}
        all_jobs = set(job.name for job in pipeline.jobs)
        
        stages = []
        scheduled = set()
        
        while scheduled != all_jobs:
            stage = []
            for job_name in all_jobs - scheduled:
                if deps[job_name] <= scheduled:
                    stage.append(job_name)
            
            if not stage:
                remaining = all_jobs - scheduled
                stage = list(remaining)
                scheduled = all_jobs
            else:
                scheduled.update(stage)
            
            stages.append(sorted(stage))
        
        return stages
    
    def generate_mermaid_pipeline_diagram(self, pipeline: Pipeline) -> str:
        """Generate a Mermaid flowchart diagram for the pipeline."""
        lines = ["```mermaid", "flowchart TD"]
        
        node_ids = {}
        for i, job in enumerate(pipeline.jobs):
            node_id = f"job{i}"
            node_ids[job.name] = node_id
            display_name = job.name
            if len(display_name) > 40:
                display_name = display_name[:37] + "..."
            lines.append(f'    {node_id}["{display_name}"]')
        
        for job in pipeline.jobs:
            if job.dependencies:
                for dep in job.dependencies:
                    if dep in node_ids:
                        lines.append(f"    {node_ids[dep]} --> {node_ids[job.name]}")
        
        root_jobs = [j for j in pipeline.jobs if not j.dependencies]
        if root_jobs:
            lines.append('    start((Start))')
            for job in root_jobs:
                lines.append(f"    start --> {node_ids[job.name]}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def generate_mermaid_job_diagram(self, job_name: str, as_image: bool = False) -> str:
        """Generate a diagram showing job structure with playbooks."""
        chain = self.resolve_job_inheritance(job_name)
        playbooks = self.get_all_playbooks_for_job(job_name)
        
        lines = ["flowchart TB"]
        
        # Job inheritance
        lines.append("    subgraph inheritance[\"Job Inheritance\"]")
        for i, job in enumerate(chain):
            node_id = f"j{i}"
            display = job.name[:25] + "..." if len(job.name) > 25 else job.name
            style = ":::highlight" if i == 0 else ""
            lines.append(f'        {node_id}["{display}"]{style}')
            if i > 0:
                lines.append(f"        j{i-1} -.->|parent| {node_id}")
        lines.append("    end")
        
        # Playbooks
        pb_sections = [
            ('pre', 'Pre-run', playbooks['pre_run']),
            ('run', 'Run', playbooks['run']),
            ('post', 'Post-run', playbooks['post_run']),
        ]
        
        lines.append("    subgraph playbooks[\"Playbooks\"]")
        for prefix, label, pb_list in pb_sections:
            if pb_list:
                lines.append(f"        subgraph {prefix}[\"{label}\"]")
                for i, pb in enumerate(pb_list):
                    pb_name = Path(pb).name[:20]
                    lines.append(f'            {prefix}{i}["{pb_name}"]')
                lines.append("        end")
        lines.append("    end")
        
        lines.append("    classDef highlight fill:#f96,stroke:#333,stroke-width:2px")
        
        mermaid_code = "\n".join(lines)
        
        # Return as image or code block
        if as_image and self.image_generator:
            return self.image_generator.mermaid_to_markdown(mermaid_code, f"Job {job_name}")
        else:
            return f"```mermaid\n{mermaid_code}\n```"
    
    def generate_deepwiki_json(self, pipeline_name: str) -> dict:
        """Generate DeepWiki-compatible JSON structure."""
        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            return {"error": f"Pipeline '{pipeline_name}' not found"}
        
        # Generate pages
        pages = []
        
        # Overview page
        overview_content = self._generate_overview_page(pipeline)
        pages.append({
            "id": "1-overview",
            "title": "Pipeline Overview",
            "content": overview_content,
            "importance": 10,
            "file_hashes": [],
            "parent": None
        })
        
        # Architecture page
        arch_content = self._generate_architecture_page(pipeline)
        pages.append({
            "id": "2-architecture",
            "title": "Pipeline Architecture",
            "content": arch_content,
            "importance": 9,
            "file_hashes": [],
            "parent": None
        })
        
        # Job details pages
        for i, pipeline_job in enumerate(pipeline.jobs):
            job_content = self._generate_job_page(pipeline_job, pipeline)
            pages.append({
                "id": f"3.{i+1}-job-{self._slugify(pipeline_job.name)}",
                "title": f"Job: {pipeline_job.name}",
                "content": job_content,
                "importance": 7,
                "file_hashes": [],
                "parent": "2-architecture"
            })
        
        # Playbooks page
        playbooks_content = self._generate_playbooks_page(pipeline)
        pages.append({
            "id": "4-playbooks",
            "title": "Playbook Reference",
            "content": playbooks_content,
            "importance": 6,
            "file_hashes": [],
            "parent": None
        })
        
        # Generate cache key (DeepWiki format)
        cache_key = hashlib.md5(
            f"zuul_{pipeline_name}_main_en".encode()
        ).hexdigest()
        
        return {
            "id": cache_key,
            "repo_url": str(self.repo_path),
            "branch": "main",
            "language": "en",
            "wiki_type": "comprehensive",
            "generated_at": datetime.now().isoformat(),
            "analyzer": "zuul-deepwiki-analyzer",
            "version": "1.0.0",
            "structure": {
                "pages": pages
            },
            "content": {page["id"]: page["content"] for page in pages}
        }
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug."""
        return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:50]
    
    def _generate_overview_page(self, pipeline: Pipeline) -> str:
        """Generate the overview page content."""
        execution_order = self._calculate_execution_order(pipeline)
        
        # Get AI summary if available
        job_summaries = {}
        if self.ai.enabled:
            for pj in pipeline.jobs:
                job = self.jobs.get(pj.name)
                if job:
                    playbooks = self.get_all_playbooks_for_job(pj.name)
                    pb_summaries = [
                        self.parse_playbook(pb)
                        for pb in playbooks['pre_run'] + playbooks['run'] + playbooks['post_run']
                    ]
                    job_summaries[pj.name] = self.ai.summarize_job(job, pb_summaries)
        
        pipeline_summary = ""
        if self.ai.enabled:
            pipeline_summary = self.ai.summarize_pipeline(pipeline, job_summaries)
        
        content = f"""# {pipeline.name}

## Overview

{pipeline_summary if pipeline_summary else "This pipeline orchestrates a series of CI/CD jobs for testing and deployment."}

**Total Jobs:** {len(pipeline.jobs)}  
**Execution Stages:** {len(execution_order)}

## Execution Stages

The pipeline executes jobs in the following order, with jobs in each stage running in parallel:

"""
        for i, stage in enumerate(execution_order):
            content += f"### Stage {i + 1}\n\n"
            for job_name in stage:
                summary = job_summaries.get(job_name, "")
                content += f"- **{job_name}**"
                if summary:
                    content += f": {summary}"
                content += "\n"
            content += "\n"
        
        return content
    
    def _generate_architecture_page(self, pipeline: Pipeline) -> str:
        """Generate the architecture page content."""
        content = f"""# Pipeline Architecture

## Job Inheritance Structure

Zuul jobs use inheritance to share configuration. Here's how jobs in this pipeline inherit from parent jobs:

"""
        for pj in pipeline.jobs:
            chain = self.resolve_job_inheritance(pj.name)
            if len(chain) > 1:
                chain_str = " → ".join(f"`{j.name}`" for j in chain)
                content += f"- **{pj.name}**: {chain_str}\n"
        
        content += """

## Playbook Execution Flow

Each job executes playbooks in three phases:

1. **Pre-run**: Setup and preparation tasks (inherited from parent, runs before main playbooks)
2. **Run**: Main execution playbooks (overrides parent definition)
3. **Post-run**: Cleanup and log collection (runs after main playbooks, even on failure)

"""
        return content
    
    def _generate_job_page(self, pipeline_job: PipelineJob, pipeline: Pipeline) -> str:
        """Generate a detailed page for a single job."""
        job = self.jobs.get(pipeline_job.name)
        playbooks = self.get_all_playbooks_for_job(pipeline_job.name)
        
        content = f"""# {pipeline_job.name}

"""
        if job:
            if job.description:
                content += f"## Description\n\n{job.description}\n\n"
            
            if job.ai_summary:
                content += f"## Summary\n\n{job.ai_summary}\n\n"
            
            # Inheritance
            chain = self.resolve_job_inheritance(pipeline_job.name)
            if len(chain) > 1:
                content += "## Job Inheritance\n\n"
                content += f"{self.generate_mermaid_job_diagram(pipeline_job.name, as_image=self.generate_images)}\n\n"
                content += "**Inheritance Chain:** " + " → ".join(f"`{j.name}`" for j in chain) + "\n\n"
            
            # Configuration
            content += "## Configuration\n\n"
            content += f"| Property | Value |\n|----------|-------|\n"
            if job.nodeset:
                content += f"| Nodeset | `{job.nodeset}` |\n"
            if job.timeout:
                content += f"| Timeout | {job.timeout}s ({job.timeout // 3600}h {(job.timeout % 3600) // 60}m) |\n"
            content += f"| Source File | `{job.source_file}` |\n"
            content += "\n"
            
            # Dependencies
            if pipeline_job.dependencies:
                content += "## Dependencies\n\nThis job requires the following jobs to complete first:\n\n"
                for dep in pipeline_job.dependencies:
                    content += f"- `{dep}`\n"
                content += "\n"
            
            # Playbooks
            content += "## Playbooks\n\n"
            
            for phase, pb_list in [('Pre-run', playbooks['pre_run']), 
                                    ('Run', playbooks['run']), 
                                    ('Post-run', playbooks['post_run'])]:
                if pb_list:
                    content += f"### {phase}\n\n"
                    for pb in pb_list:
                        summary = self.parse_playbook(pb)
                        content += f"#### `{pb}`\n\n"
                        if summary.ai_summary:
                            content += f"{summary.ai_summary}\n\n"
                        elif summary.description:
                            content += f"*{summary.description}*\n\n"
                        
                        if summary.hosts:
                            content += f"**Hosts:** `{summary.hosts}`\n\n"
                        
                        if summary.tasks:
                            content += "**Tasks:**\n"
                            for task in summary.tasks:
                                content += f"- {task}\n"
                            content += "\n"
            
            # Variables
            if job.vars:
                content += "## Variables\n\n"
                content += "| Variable | Value |\n|----------|-------|\n"
                for var_name, var_value in list(job.vars.items())[:15]:
                    value_str = str(var_value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    value_str = value_str.replace("|", "\\|").replace("\n", " ")
                    content += f"| `{var_name}` | `{value_str}` |\n"
                if len(job.vars) > 15:
                    content += f"| ... | *{len(job.vars) - 15} more variables* |\n"
                content += "\n"
        else:
            content += "⚠️ *Job definition not found in this repository (may be defined externally)*\n\n"
        
        return content
    
    def _generate_playbooks_page(self, pipeline: Pipeline) -> str:
        """Generate a reference page for all playbooks."""
        all_playbooks = set()
        for pj in pipeline.jobs:
            playbooks = self.get_all_playbooks_for_job(pj.name)
            all_playbooks.update(playbooks['pre_run'])
            all_playbooks.update(playbooks['run'])
            all_playbooks.update(playbooks['post_run'])
        
        content = """# Playbook Reference

This page provides a reference for all Ansible playbooks used in this pipeline.

## Playbook Index

| Playbook | Description | Hosts |
|----------|-------------|-------|
"""
        for pb in sorted(all_playbooks):
            summary = self.parse_playbook(pb)
            desc = (summary.ai_summary or summary.description or "N/A")[:60]
            if len(desc) == 60:
                desc += "..."
            hosts = summary.hosts or "N/A"
            content += f"| `{pb}` | {desc} | `{hosts}` |\n"
        
        content += "\n## Detailed Playbook Information\n\n"
        
        for pb in sorted(all_playbooks):
            summary = self.parse_playbook(pb)
            content += f"### `{pb}`\n\n"
            
            if not summary.exists:
                content += f"⚠️ {summary.error}\n\n"
                continue
            
            if summary.ai_summary:
                content += f"{summary.ai_summary}\n\n"
            elif summary.description:
                content += f"*{summary.description}*\n\n"
            
            if summary.hosts:
                content += f"**Hosts:** `{summary.hosts}`\n\n"
            
            if summary.roles:
                content += f"**Roles:** {', '.join(f'`{r}`' for r in summary.roles)}\n\n"
            
            if summary.tasks:
                content += "**Tasks:**\n"
                for task in summary.tasks:
                    content += f"- {task}\n"
                content += "\n"
            
            content += "---\n\n"
        
        return content
    
    def generate_markdown_report(self, pipeline_name: str) -> str:
        """Generate a comprehensive Markdown report for a pipeline."""
        pipeline = self.pipelines.get(pipeline_name)
        
        if not pipeline:
            matches = [name for name in self.pipelines.keys() if pipeline_name in name]
            if matches:
                return f"Pipeline '{pipeline_name}' not found. Did you mean one of these?\n" + \
                       "\n".join(f"  - {m}" for m in matches)
            
            available = "\n".join(f"  - {name}" for name in sorted(self.pipelines.keys())[:20])
            return f"Pipeline '{pipeline_name}' not found.\n\nAvailable pipelines:\n{available}"
        
        # Generate all pages and combine into single markdown
        deepwiki_data = self.generate_deepwiki_json(pipeline_name)
        
        content = ""
        for page in deepwiki_data["structure"]["pages"]:
            content += deepwiki_data["content"][page["id"]]
            content += "\n\n---\n\n"
        
        return content
    
    def list_pipelines(self) -> str:
        """List all available pipelines."""
        lines = ["# Available Pipelines", ""]
        
        for name in sorted(self.pipelines.keys()):
            pipeline = self.pipelines[name]
            lines.append(f"- **{name}** ({len(pipeline.jobs)} jobs)")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Zuul CI/CD pipelines and generate DeepWiki-compatible documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./my-repo my-pipeline-name
  %(prog)s ./my-repo my-pipeline --format deepwiki -o wiki.json
  %(prog)s ./my-repo my-pipeline --ai-enhance
  %(prog)s ./my-repo --list
        """
    )
    
    parser.add_argument(
        "repo_path",
        help="Path to the repository containing Zuul configurations"
    )
    
    parser.add_argument(
        "pipeline_name",
        nargs="?",
        help="Name of the pipeline to analyze"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available pipelines"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["markdown", "deepwiki"],
        default="markdown",
        help="Output format: 'markdown' for readable docs, 'deepwiki' for JSON import"
    )
    
    parser.add_argument(
        "--ai-enhance",
        action="store_true",
        help="Use AI to enhance documentation (requires ANTHROPIC_API_KEY env var)"
    )
    
    parser.add_argument(
        "--deepwiki-url",
        help="URL of running DeepWiki instance for AI features (e.g., http://localhost:8001)"
    )
    
    parser.add_argument(
        "--images",
        action="store_true",
        help="Generate image links instead of Mermaid code blocks (uses mermaid.ink service)"
    )
    
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download images as SVG files (requires --images, needs httpx)"
    )
    
    args = parser.parse_args()
    
    # Validate repo path
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Setup AI enhancer if requested
    ai_enhancer = None
    if args.ai_enhance or args.deepwiki_url:
        ai_enhancer = AIEnhancer(
            deepwiki_url=args.deepwiki_url
        )
        if ai_enhancer.enabled:
            print("AI enhancement enabled", file=sys.stderr)
        else:
            print("Warning: AI enhancement requested but not available", file=sys.stderr)
    
    # Create analyzer and parse
    analyzer = ZuulDeepWikiAnalyzer(
        str(repo_path), 
        ai_enhancer,
        output_dir=args.output,
        generate_images=args.images,
        download_images=args.download_images
    )
    
    if args.images:
        print("Image generation enabled - using mermaid.ink URLs for diagrams", file=sys.stderr)
        if args.download_images:
            if HTTPX_AVAILABLE:
                print("Will attempt to download images as SVG files", file=sys.stderr)
            else:
                print("Warning: httpx not installed, images will use URLs only", file=sys.stderr)
                print("Install with: pip install httpx", file=sys.stderr)
    
    print("Parsing Zuul configuration files...", file=sys.stderr)
    analyzer.parse_zuul_files()
    print(f"Found {len(analyzer.jobs)} jobs and {len(analyzer.pipelines)} pipelines", file=sys.stderr)
    
    # Generate output
    if args.list:
        output = analyzer.list_pipelines()
    elif args.pipeline_name:
        if args.format == "deepwiki":
            output = json.dumps(analyzer.generate_deepwiki_json(args.pipeline_name), indent=2)
        else:
            output = analyzer.generate_markdown_report(args.pipeline_name)
    else:
        print("Error: Please specify a pipeline name or use --list", file=sys.stderr)
        sys.exit(1)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
