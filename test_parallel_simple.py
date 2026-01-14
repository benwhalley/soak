#!/usr/bin/env python3
"""Test if Map nodes send parallel API requests."""
import asyncio
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Patch litellm to log timing BEFORE importing soak
import litellm
_original_completion = litellm.completion
call_times = []

def logged_completion(*args, **kwargs):
    start = time.time()
    logger.info(f"🚀 API call START")
    call_times.append(('start', start))
    result = _original_completion(*args, **kwargs)
    end = time.time()
    logger.info(f"✅ API call DONE ({end-start:.2f}s)")
    call_times.append(('end', end))
    return result

litellm.completion = logged_completion

# Now import and run soak
from soak.models.dag import DAG, DAGConfig
from soak.models.nodes.map import Map
from soak.models.base import TrackedItem
import anyio

# Create simple DAG with Map node
dag = DAG(name="test")
dag.config = DAGConfig(
    documents=[TrackedItem(content=f"item_{i}", id=f"item_{i}", sources=[f"item_{i}"], metadata={})
               for i in range(5)],
    model_name="gpt-4o-mini",
    show_progress=False,
)

map_node = Map(
    name="test_map",
    inputs=["documents"],
    template="Say hello to {{input}}.\n\n[[greeting:str]]"
)
dag.add_node(map_node)

logger.info("="*60)
logger.info("Running Map with 5 items")
logger.info("If parallel: all START logs before DONE logs")
logger.info("If sequential: START/DONE pairs alternate")
logger.info("="*60)

# Run
result, error = anyio.run(dag.run)

logger.info("="*60)
logger.info("Complete!")

# Analyze timing
if len(call_times) >= 10:
    starts = [t for label, t in call_times if label == 'start']
    ends = [t for label, t in call_times if label == 'end']

    # Check if requests overlapped
    first_start = min(starts)
    last_start = max(starts)
    first_end = min(ends)

    if last_start < first_end:
        logger.info("✅ PARALLEL: Last request started before first request finished")
        logger.info(f"   Start spread: {last_start - first_start:.2f}s")
        logger.info(f"   First request finished at: {first_end - first_start:.2f}s")
    else:
        logger.warning("⚠️  SEQUENTIAL: All starts came after all ends")
