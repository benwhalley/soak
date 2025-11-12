# Hybrid Template System

Soak supports both **inline templates** (within .soak files) and **external template files** (.sd files), giving you flexibility in how you organize your pipelines.

## Quick Start

### Inline Templates (Current Approach)

The traditional approach -- define templates directly in your .soak file:

```yaml
name: my_pipeline
nodes:
  - name: summarize
    type: Map

---#summarize
Summarize this text: {{input}}
[[summary]]
```

### External Templates (New Approach)

Split templates into separate .sd files for reusability:

**pipeline.soak:**
```yaml
name: my_pipeline
nodes:
  - name: summarize
    type: Map
```

**summarize.sd:**
```
Summarize this text: {{input}}
[[summary]]
```

## Template Resolution

### Priority Order

When loading a node template, soak searches in this order:

1. **Inline template** (`---#node_name` in .soak file) -- highest priority
2. **Explicit template field** (`template: filename.sd` in YAML)
3. **Convention** (looks for `{node_name}.sd` in search paths)

### Search Paths

External templates are searched in this order:

1. **Current directory** (where .soak file is)
2. **templates/ subdirectory** (in same directory as .soak file)
3. **Custom directories** (specified via `template_dirs` in YAML)
4. **Package templates** (`soak/templates/` -- for defaults)

### Examples

#### Convention-based Loading

Create `summarize.sd` next to your pipeline:

```
project/
  pipeline.soak
  summarize.sd  ← Found by convention (node name matches)
```

**pipeline.soak:**
```yaml
name: my_pipeline
nodes:
  - name: summarize
    type: Map
```

#### Explicit Template Files

Specify a different filename:

```yaml
name: my_pipeline
nodes:
  - name: summarize
    type: Map
    template: custom_summary.sd  # explicit filename
```

#### Templates Subdirectory

Organize templates in a subdirectory:

```
project/
  pipeline.soak
  templates/
    summarize.sd
    classify.sd
```

Templates are automatically found in `templates/` subdirectory.

#### Custom Template Directories

Specify additional search directories:

```yaml
name: my_pipeline
template_dirs:
  - /path/to/shared/templates
  - ~/my_templates
nodes:
  - name: summarize
    type: Map
```

## Use Cases

### 1. Template Reuse Across Pipelines

Create a library of reusable templates:

```
templates/
  summarize.sd
  extract_themes.sd
  classify_sentiment.sd

pipeline1.soak  # uses summarize.sd
pipeline2.soak  # also uses summarize.sd
```

### 2. Large Pipelines

Split complex pipelines for readability:

```
project/
  cfs_analysis.soak  # 50 lines of YAML structure
  templates/
    filter_relevant.sd
    extract_codes.sd
    generate_themes.sd
    write_narrative.sd
```

### 3. Version Control

External templates make diffs cleaner:
- Changes to pipeline structure (YAML) separate from prompt changes (.sd files)
- Easier to review prompt modifications

### 4. Mixed Approach

Use both approaches in the same pipeline:

```yaml
name: my_pipeline
nodes:
  - name: filter
    type: Filter
    template: shared_filter.sd  # external, reusable

  - name: custom_analysis
    type: Transform
    # inline for one-off logic

---#custom_analysis
This is a custom, pipeline-specific template.
[[result]]
```

## Migration Guide

### From Inline to External

**Before:**
```yaml
name: pipeline
nodes:
  - name: summarize
    type: Map

---#summarize
Summarize: {{input}}
[[summary]]
```

**After:**

1. Create `summarize.sd`:
   ```
   Summarize: {{input}}
   [[summary]]
   ```

2. Remove inline template from .soak:
   ```yaml
   name: pipeline
   nodes:
     - name: summarize
       type: Map
   ```

That's it! The template is automatically found by convention.

## Best Practices

1. **Use inline templates for:**
   - One-off, pipeline-specific logic
   - Small pipelines (< 200 lines total)
   - Quick prototyping

2. **Use external templates for:**
   - Reusable prompts across projects
   - Large pipelines (> 500 lines)
   - Team collaboration (cleaner diffs)

3. **Organize templates:**
   - Put shared templates in `templates/` subdirectory
   - Use descriptive filenames (`extract_recovery_codes.sd` not `step1.sd`)
   - Keep templates close to .soak files (avoid deep nesting)

4. **Version control:**
   - Commit both .soak and .sd files
   - Use meaningful commit messages when changing prompts
   - Consider separate PR reviews for structure vs. prompt changes

## Error Messages

If a template cannot be found, you'll get a helpful error:

```
Template 'summarize.sd' not found.

Searched in:
  - /path/to/project/summarize.sd
  - /path/to/project/templates/summarize.sd
  - /path/to/soak/templates/summarize.sd

Suggestions:
  - Add an inline template with ---#summarize
  - Create summarize.sd in one of the search paths
  - Specify a different template file in the node config
```

## Technical Details

- Template files use UTF-8 encoding
- Search paths are resolved at pipeline load time
- Duplicate paths are automatically deduplicated
- Invalid paths in `template_dirs` are logged as warnings
- Template resolution is lazy (only loads when needed)
