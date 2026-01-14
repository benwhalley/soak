#!/usr/bin/env python3
"""Test if Map nodes actually send parallel API requests.

This script creates a simple pipeline and logs timestamps of API calls
to verify they're happening concurrently.
"""
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

# Configure logging to show timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Patch litellm.completion to log timing
import litellm
_original_completion = litellm.completion

def logged_completion(*args, **kwargs):
    start = time.time()
    logger.info(f"🚀 API call START (thread {id(asyncio.current_task())})")
    result = _original_completion(*args, **kwargs)
    elapsed = time.time() - start
    logger.info(f"✅ API call DONE in {elapsed:.2f}s")
    return result

litellm.completion = logged_completion

# Now run a simple test
from soak.specs import load_template_bundle

# Create a minimal test pipeline
test_spec = """
name: parallel_test
default_context:
  num_items: 5

nodes:
  - name: test_map
    type: Map
    inputs: [documents]
---#test_map
Tell me a joke about {{input}}.

[[joke:str]]
"""

# Write spec to temp file
spec_path = Path("/tmp/test_parallel.yaml")
spec_path.write_text(test_spec)

# Create test documents
test_docs = [f"topic_{i}" for i in range(5)]

# Load and run pipeline
pipeline = load_template_bundle(spec_path)
dag = pipeline.dag
dag.config.documents = test_docs
dag.config.show_progress = False
dag.config.model_name = "gpt-4o-mini"

logger.info("="*60)
logger.info("Starting pipeline with 5 items in Map node")
logger.info("If parallel: all 5 START logs should appear before DONE logs")
logger.info("If sequential: START/DONE pairs should alternate")
logger.info("="*60)

# Run the pipeline
import anyio
result, error = anyio.run(dag.run)

logger.info("="*60)
logger.info("Pipeline complete!")
if error:
    logger.error(f"Error: {error}")
