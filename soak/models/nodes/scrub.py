"""Scrub node for PII detection and redaction using scrubadub."""

import fnmatch
import hashlib
import importlib
import inspect
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from joblib import Memory
from pydantic import Field, PrivateAttr

from soak.models.base import TrackedItem
from soak.models.text_utils import escape_struckdown_chars

from .base import ItemsNode

logger = logging.getLogger(__name__)

# suppress scrubadub locale warnings globally for this module
warnings.filterwarnings("ignore", message=".*does not support the locale.*", category=UserWarning)

# cache for scrubadub detector results
scrubadub_memory = Memory(Path(".scrubadub_cache"), verbose=0)


import spacy
from spacy.cli import download as spacy_download

def download_model(model_name):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.warning(f"SpaCy model '{model_name}' not found. Downloading now...")
        try:
            spacy_download(model_name, False)  # False = don't force reinstall if cached
            nlp = spacy.load(model_name)
            logger.info(f"✓ SpaCy model '{model_name}' downloaded and loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download SpaCy model '{model_name}': {e}")


@scrubadub_memory.cache
def _cached_detect_filth(
    text: str,
    detector_config_hash: str,
    detectors_tuple: Tuple[str, ...],
    spacy_model: str,
    locale: str,
    exclude_detectors_tuple: Tuple[str, ...]
) -> List[Dict[str, Any]]:
    """Cached filth detection function.

    This function performs the expensive scrubadub detection and is cached
    based on text content and detector configuration.

    Args:
        text: Text to scan for PII
        detector_config_hash: Hash of detector configuration (for cache key)
        detectors_tuple: Tuple of detector patterns (immutable for caching)
        spacy_model: SpaCy model name
        locale: Locale string
        exclude_detectors_tuple: Tuple of excluded detector names

    Returns:
        List of filth detection dicts with positions and types
    """
    import scrubadub

    logger.debug(f"Cache miss for text hash {hashlib.md5(text.encode()).hexdigest()[:8]}... running detector")

    # resolve detectors (this will be cached indirectly via the function cache)
    detectors_list = list(detectors_tuple)
    exclude_list = list(exclude_detectors_tuple)

    # instantiate detectors
    resolved_detectors = []
    for pattern in detectors_list:
        if "*" in pattern:
            matched = _resolve_glob_pattern_static(pattern, locale, spacy_model)
            resolved_detectors.extend(matched)
        else:
            detector_cls = _import_detector_class_static(pattern, locale, spacy_model)
            if detector_cls:
                resolved_detectors.append(detector_cls)

    # deduplicate and filter excluded
    fresh_detectors = []
    seen_detector_names = set()

    for detector in resolved_detectors:
        inst = None
        try:
            if inspect.isclass(detector):
                # instantiate
                if "Spacy" in detector.__name__ or "spacy" in detector.__name__:
                    try:
                        nlp = spacy.load(spacy_model)
                        inst = detector(nlp=nlp, locale=locale)
                    except Exception as e:
                        logger.debug(f"Failed to load SpaCy model for {detector.__name__}: {e}")
                        continue
                else:
                    try:
                        inst = detector(locale=locale)
                    except TypeError:
                        try:
                            inst = detector()
                        except Exception as e:
                            # some detectors need special args we can't provide - skip them
                            logger.debug(f"Failed to instantiate {detector.__name__}: {e}")
                            continue
            else:
                # already an instance, create fresh copy
                try:
                    if "Spacy" in detector.__class__.__name__ or "spacy" in detector.__class__.__name__:
                        nlp = spacy.load(spacy_model)
                        inst = detector.__class__(nlp=nlp, locale=locale)
                    else:
                        try:
                            inst = detector.__class__(locale=locale)
                        except TypeError:
                            inst = detector.__class__()
                except Exception as e:
                    logger.debug(f"Failed to create fresh detector from {detector}: {e}")
                    continue

            if inst and inst.name not in seen_detector_names and inst.name not in exclude_list:
                fresh_detectors.append(inst)
                seen_detector_names.add(inst.name)
        except Exception as e:
            logger.debug(f"Failed to process detector {detector}: {e}")
            continue

    # create scrubber and detect filth
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        scrubber = scrubadub.Scrubber(detector_list=fresh_detectors, locale=locale)

    filth_items = list(scrubber.iter_filth(text))

    # convert to serializable dicts (without source_id, which will be added later)
    filth_dicts = []
    for filth in filth_items:
        filth_dicts.append({
            "type": filth.type,
            "text": filth.text,
            "beg": filth.beg,
            "end": filth.end,
        })

    return filth_dicts


def _resolve_glob_pattern_static(pattern: str, locale: str, spacy_model: str) -> List[Any]:
    """Static version of glob pattern resolution for caching."""
    import scrubadub

    parts = pattern.rsplit(".", 1)
    if len(parts) != 2:
        return []

    module_path, class_pattern = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return []

    matched = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if hasattr(scrubadub.detectors, "Detector"):
            base_detector = scrubadub.detectors.Detector
            base_class_names = {"Detector", "RegexDetector", "KnownFilthDetector"}

            if (issubclass(obj, base_detector) and
                obj is not base_detector and
                obj.__name__ not in base_class_names):
                if fnmatch.fnmatch(name, class_pattern):
                    matched.append(obj)

    return matched


def _import_detector_class_static(class_path: str, locale: str, spacy_model: str) -> Optional[Any]:
    """Static version of detector class import for caching."""
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        return None

    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
        detector_cls = getattr(module, class_name)
        return detector_cls
    except Exception:
        return None


class Scrub(ItemsNode):
    """Node for detecting and redacting PII using scrubadub.

    Supports:
    - Glob patterns for detector selection (e.g., "scrubadub.detectors.*")
    - Third-party detectors (scrubadub_spacy, scrubadub_address)
    - Configurable SpaCy model for name detection
    - Detailed PII logging with context
    - Statistics tracking for HTML display
    """

    type: Literal["Scrub"] = "Scrub"

    detectors: List[str] = Field(
        default_factory=lambda: ["scrubadub.detectors.*", "scrubadub_spacy.detectors.SpacyNameDetector"],
        description="List of detector patterns (supports glob). Examples: 'scrubadub.detectors.*', 'scrubadub_spacy.detectors.SpacyNameDetector'",
    )

    spacy_model: str = Field(
        default="en_core_web_trf",
        description="SpaCy model to use for name detection (e.g., en_core_web_trf, en_core_web_sm)",
    )

    redact: bool = Field(
        default=True,
        description="If True, replace detected PII with placeholders. If False, just detect and log.",
    )

    log_filth: bool = Field(
        default=True,
        description="If True, write FILTH.md with detailed PII detection log to export folder",
    )

    locale: str = Field(
        default="en_GB",
        description="Locale for detector configuration (e.g., en_GB, en_US)",
    )

    exclude_detectors: List[str] = Field(
        default_factory=lambda: [
            "unknown",
            "tagged_evaluation"  # requires known_filth_items arg
        ],
        description="List of detector names to exclude (by detector.name, not class name)",
    )

    # private attributes for statistics (not serialized)
    _total_items: int = PrivateAttr(default=0)
    _total_filth_detected: int = PrivateAttr(default=0)
    _items_with_filth: int = PrivateAttr(default=0)
    _filth_by_type: Dict[str, int] = PrivateAttr(default_factory=dict)
    _filth_log: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _resolved_detectors: Optional[List[Any]] = PrivateAttr(default=None)
    _detector_warnings: List[str] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # resolve detectors during initialization
        # suppress locale warnings from scrubadub and capture them
        import warnings
        with warnings.catch_warnings(record=True) as w:
            # suppress warnings to user but record them for logging
            warnings.simplefilter("always")  # capture all warnings
            warnings.filterwarnings("ignore", category=UserWarning)  # don't print to stderr
            try:
                self._resolved_detectors = self._resolve_detectors()
                # capture locale warnings for export (but skip excluded detectors)
                for warning in w:
                    msg = str(warning.message)
                    if "does not support the locale" in msg:
                        # don't log warnings for detectors we're excluding anyway
                        is_excluded = any(det_name in msg for det_name in self.exclude_detectors)
                        if not is_excluded:
                            self._detector_warnings.append(msg)
                            logger.debug(msg)
            except ImportError as e:
                logger.error(f"Failed to resolve detectors: {e}")
                logger.error("Install scrubadub with: pip install 'soak[scrub]'")
                raise

    def _resolve_detectors(self) -> List[Any]:
        """Resolve detector patterns to actual detector classes.

        Supports:
        - Glob patterns: "scrubadub.detectors.*" matches all detectors in module
        - Specific classes: "scrubadub.detectors.PhoneDetector"
        - Third-party: "scrubadub_spacy.detectors.SpacyNameDetector"

        Returns:
            List of detector class instances ready to use
        """
        try:
            import scrubadub
        except ImportError:
            raise ImportError(
                "scrubadub is required for Scrub node. "
                "Install with: pip install scrubadub or pip install 'soak[scrub]'"
            )

        resolved = []

        for pattern in self.detectors:
            # check if pattern contains glob
            if "*" in pattern:
                # glob pattern: introspect module to find matching classes
                matched = self._resolve_glob_pattern(pattern)
                resolved.extend(matched)
            else:
                # specific class path
                detector_cls = self._import_detector_class(pattern)
                if detector_cls:
                    resolved.append(detector_cls)

        # deduplicate by class name
        seen = set()
        unique_detectors = []
        for det in resolved:
            cls_name = det.__class__.__name__ if not inspect.isclass(det) else det.__name__
            if cls_name not in seen:
                seen.add(cls_name)
                unique_detectors.append(det)

        logger.debug(f"Resolved {len(unique_detectors)} detectors: {[d.__class__.__name__ if not inspect.isclass(d) else d.__name__ for d in unique_detectors]}")

        return unique_detectors

    def _resolve_glob_pattern(self, pattern: str) -> List[Any]:
        """Resolve a glob pattern to matching detector classes.

        Args:
            pattern: Glob pattern like "scrubadub.detectors.*"

        Returns:
            List of detector class instances
        """
        try:
            import scrubadub
        except ImportError:
            return []

        # parse pattern: "scrubadub.detectors.*" -> module="scrubadub.detectors", pattern="*"
        parts = pattern.rsplit(".", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid glob pattern: {pattern}")
            return []

        module_path, class_pattern = parts

        try:
            # import the module
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Could not import module '{module_path}': {e}")
            return []

        # find all classes in module that match pattern
        matched = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # check if this is a Detector subclass (not the base class itself)
            if hasattr(scrubadub.detectors, "Detector"):
                base_detector = scrubadub.detectors.Detector
                # exclude base classes that shouldn't be instantiated directly
                base_class_names = {"Detector", "RegexDetector", "KnownFilthDetector"}

                if (issubclass(obj, base_detector) and
                    obj is not base_detector and
                    obj.__name__ not in base_class_names):
                    # check if name matches glob pattern
                    if fnmatch.fnmatch(name, class_pattern):
                        try:
                            # instantiate detector
                            detector = self._instantiate_detector(obj)
                            if detector:
                                matched.append(detector)
                        except Exception as e:
                            logger.warning(f"Failed to instantiate {name}: {e}")

        return matched

    def _import_detector_class(self, class_path: str) -> Optional[Any]:
        """Import and instantiate a specific detector class.

        Args:
            class_path: Full class path like "scrubadub.detectors.PhoneDetector"

        Returns:
            Detector instance or None if failed
        """
        # parse class path
        parts = class_path.rsplit(".", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid class path: {class_path}")
            return None

        module_path, class_name = parts

        try:
            # import module
            module = importlib.import_module(module_path)
            # get class
            detector_cls = getattr(module, class_name)
            # instantiate
            return self._instantiate_detector(detector_cls)
        except ImportError as e:
            logger.warning(f"Could not import '{class_path}': {e}")
            logger.warning("For third-party detectors, ensure the package is installed")
            return None
        except AttributeError as e:
            logger.warning(f"Class '{class_name}' not found in module '{module_path}': {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to instantiate '{class_path}': {e}")
            return None

    def _instantiate_detector(self, detector_cls: type) -> Optional[Any]:
        """Instantiate a detector class, handling special cases.

        Args:
            detector_cls: Detector class to instantiate

        Returns:
            Detector instance or None if failed
        """
        class_name = detector_cls.__name__

        # special handling for SpaCy detector
        if "Spacy" in class_name or "spacy" in class_name:
            download_model(self.spacy_model)

        # default instantiation with locale
        try:
            return detector_cls(locale=self.locale)
        except TypeError:
            # some detectors might not accept locale parameter
            try:
                return detector_cls()
            except Exception as e:
                logger.debug(f"Failed to instantiate {class_name}: {e}")
                return None
        except Exception as e:
            logger.debug(f"Failed to instantiate {class_name} with locale: {e}")
            return None

    def _get_config_hash(self) -> str:
        """Generate hash of detector configuration for cache key."""
        config_str = (
            f"{sorted(self.detectors)}"
            f"{self.spacy_model}"
            f"{self.locale}"
            f"{sorted(self.exclude_detectors)}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    async def process_items(
        self, items: List[Any], progress_bar: Optional[Any] = None
    ) -> List[TrackedItem]:
        """Process items through scrubadub.

        Args:
            items: Flat list of items in this batch
            progress_bar: Optional tqdm progress bar to update

        Returns:
            List of TrackedItems with scrubbed content
        """
        if not self._resolved_detectors:
            logger.warning("No detectors available -- skipping PII detection")
            # return items unchanged
            results = []
            for item in items:
                if isinstance(item, TrackedItem):
                    results.append(item)
                else:
                    results.append(
                        TrackedItem(
                            content=str(item),
                            id="unknown",
                            sources=["unknown"],
                            metadata={}
                        )
                    )
            return results

        results = []

        for item in items:
            # extract content from TrackedItem
            if isinstance(item, TrackedItem):
                content = item.content
                source_id = item.id
                metadata = item.metadata or {}
                sources = item.sources
            else:
                # backward compatibility for plain strings
                content = str(item)
                source_id = "unknown"
                metadata = {}
                sources = ["unknown"]

            # process with scrubadub
            scrubbed_content, filth_items = self._scrub_text(content, source_id)

            # track statistics
            self._total_items += 1
            if filth_items:
                self._items_with_filth += 1
            self._total_filth_detected += len(filth_items)

            for filth_entry in filth_items:
                filth_type = filth_entry["type"]
                self._filth_by_type[filth_type] = self._filth_by_type.get(filth_type, 0) + 1

                # log for FILTH.md
                if self.log_filth:
                    self._filth_log.append(filth_entry)

            # create new TrackedItem with scrubbed content
            # escape @ symbols to prevent struckdown parsing errors
            final_content = scrubbed_content if self.redact else content
            final_content = escape_struckdown_chars(final_content)

            new_item = TrackedItem(
                content=final_content,
                id=f"{source_id}__scrubbed",  # preserve provenance
                sources=sources,
                metadata={
                    **metadata,
                    "scrubbed": self.redact,
                    "filth_detected": len(filth_items),
                    "filth_types": list(set(f["type"] for f in filth_items)),
                }
            )

            results.append(new_item)

            # update progress bar
            if progress_bar is not None:
                progress_bar.update(1)

        return results

    def _scrub_text(self, text: str, source_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process text with scrubadub and return scrubbed text + detected filth.

        Uses cached detection results when processing the same text with the same
        detector configuration, significantly improving performance on repeated runs.

        Args:
            text: Text to process
            source_id: ID of source document for logging

        Returns:
            Tuple of (scrubbed_text, list_of_filth_dicts)
        """
        try:
            import scrubadub
        except ImportError:
            logger.error(
                "scrubadub is required for Scrub node. "
                "Install with: pip install scrubadub"
            )
            raise

        # use cached detection
        config_hash = self._get_config_hash()
        filth_basic = _cached_detect_filth(
            text=text,
            detector_config_hash=config_hash,
            detectors_tuple=tuple(self.detectors),
            spacy_model=self.spacy_model,
            locale=self.locale,
            exclude_detectors_tuple=tuple(self.exclude_detectors),
        )

        # build scrubbed text if redacting
        if self.redact:
            scrubbed_text = text
            # apply redactions in reverse order to preserve positions
            for filth in sorted(filth_basic, key=lambda x: x["beg"], reverse=True):
                replacement = f"[REDACTED-{filth['type'].upper()}]"
                scrubbed_text = (
                    scrubbed_text[:filth["beg"]] +
                    replacement +
                    scrubbed_text[filth["end"]:]
                )
        else:
            scrubbed_text = text

        # add context and source_id to filth dicts
        filth_dicts = []
        for filth in filth_basic:
            start_pos = filth["beg"]
            end_pos = filth["end"]

            context_start = max(0, start_pos - 20)
            context_end = min(len(text), end_pos + 20)

            before = text[context_start:start_pos]
            after = text[end_pos:context_end]

            # add ellipsis if truncated
            if context_start > 0:
                before = "..." + before
            if context_end < len(text):
                after = after + "..."

            filth_dicts.append({
                "source_id": source_id,
                "type": filth["type"],
                "text": filth["text"],
                "position": start_pos,
                "end": end_pos,
                "context_before": before,
                "context_after": after,
            })

        return scrubbed_text, filth_dicts

    def result(self) -> Dict[str, Any]:
        """Returns dict with metadata and statistics for HTML display."""
        # get base metadata from parent
        result = super().result()

        # add Scrub-specific metadata
        result["metadata"]["detectors"] = self.detectors
        result["metadata"]["spacy_model"] = self.spacy_model
        result["metadata"]["locale"] = self.locale
        result["metadata"]["exclude_detectors"] = self.exclude_detectors
        result["metadata"]["redact"] = self.redact
        result["metadata"]["log_filth"] = self.log_filth
        result["metadata"]["total_items"] = self._total_items
        result["metadata"]["total_filth_detected"] = self._total_filth_detected
        result["metadata"]["items_with_filth"] = self._items_with_filth

        # create DataFrame for statistics display
        if self._filth_by_type:
            stats_rows = [
                {"filth_type": ftype, "count": count}
                for ftype, count in sorted(
                    self._filth_by_type.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            result["stats_df"] = pd.DataFrame(stats_rows)
        else:
            result["stats_df"] = pd.DataFrame()

        # summary statistics
        result["summary"] = {
            "total_items_processed": self._total_items,
            "items_with_filth": self._items_with_filth,
            "total_filth_instances": self._total_filth_detected,
            "filth_by_type": dict(self._filth_by_type),
            "percentage_with_pii": (
                round(100 * self._items_with_filth / self._total_items, 1)
                if self._total_items > 0
                else 0
            ),
        }

        return result

    def export(self, folder: Path, unique_id: str = ""):
        """Export Scrub node details including FILTH.md log."""
        super().export(folder, unique_id=unique_id)

        # write detectors configuration
        detectors_text = "Detectors Used:\n" + "\n".join(f"  - {d}" for d in self.detectors)
        detectors_text += f"\n\nExcluded Detectors:\n" + "\n".join(f"  - {d}" for d in self.exclude_detectors)
        detectors_text += f"\n\nLocale: {self.locale}"
        detectors_text += f"\nSpaCy Model: {self.spacy_model}"
        detectors_text += "\n\nReplacement Format: [REDACTED-TYPE]"

        # add warnings if any
        if self._detector_warnings:
            detectors_text += "\n\nDetector Warnings:\n"
            for warning in self._detector_warnings:
                detectors_text += f"  - {warning}\n"

        (folder / "detectors.txt").write_text(detectors_text)

        # write statistics CSV
        if self._filth_by_type:
            stats_df = pd.DataFrame([
                {"filth_type": ftype, "count": count}
                for ftype, count in sorted(
                    self._filth_by_type.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ])
            stats_df.to_csv(folder / "filth_stats.csv", index=False)

        # write summary statistics
        summary_text = f"""PII Detection Summary
=====================
Total items processed: {self._total_items}
Items with PII: {self._items_with_filth}
Total PII instances detected: {self._total_filth_detected}
Percentage with PII: {round(100 * self._items_with_filth / self._total_items, 1) if self._total_items > 0 else 0}%

PII by Type:
"""
        for ftype, count in sorted(self._filth_by_type.items(), key=lambda x: x[1], reverse=True):
            summary_text += f"  {ftype}: {count}\n"

        (folder / "summary.txt").write_text(summary_text)

        # write FILTH.md log if enabled
        if self.log_filth and self._filth_log:
            filth_md = self._generate_filth_markdown()
            (folder / "FILTH.md").write_text(filth_md)
            logger.info(f"✓ PII log written to {folder / 'FILTH.md'}")

        # export scrubbed outputs
        if self.output:
            from .batch import BatchList

            # flatten BatchList if needed
            if isinstance(self.output, BatchList):
                all_items = self.output.flatten_all()
            else:
                all_items = self.output

            outputs_folder = folder / "outputs"
            outputs_folder.mkdir(exist_ok=True)

            for idx, item in enumerate(all_items):
                if isinstance(item, TrackedItem):
                    output_file = outputs_folder / f"{idx:04d}_{item.safe_id}.txt"
                    output_file.write_text(item.content)

                    # export metadata
                    if item.metadata:
                        metadata_file = outputs_folder / f"{idx:04d}_{item.safe_id}_metadata.json"
                        metadata_file.write_text(item.to_json())

    def _generate_filth_markdown(self) -> str:
        """Generate markdown report of detected filth with context."""
        lines = ["# PII Detection Report\n\n"]
        lines.append(f"**Total items processed:** {self._total_items}\n\n")
        lines.append(f"**Total PII instances detected:** {self._total_filth_detected}\n\n")
        lines.append(f"**Items with PII:** {self._items_with_filth} ({round(100 * self._items_with_filth / self._total_items, 1) if self._total_items > 0 else 0}%)\n\n")

        lines.append("## PII by Type\n\n")
        for ftype, count in sorted(self._filth_by_type.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{ftype}**: {count} instances\n")
        lines.append("\n")

        # group by source_id
        by_source = {}
        for entry in self._filth_log:
            source_id = entry["source_id"]
            if source_id not in by_source:
                by_source[source_id] = []
            by_source[source_id].append(entry)

        lines.append("## Detected PII by Source\n\n")

        # write entries grouped by source
        for source_id, entries in sorted(by_source.items()):
            lines.append(f"### Source: `{source_id}`\n\n")
            lines.append(f"Found {len(entries)} PII instance(s):\n\n")

            for entry in entries:
                # format with context
                context_str = f"{entry['context_before']}**{{{{{entry['type']}}}}}**{entry['context_after']}"
                lines.append(f"- **{entry['type']}** at position {entry['position']}: `{entry['text']}`\n")
                lines.append(f"  - Context: {context_str}\n")

            lines.append("\n")

        return "".join(lines)
