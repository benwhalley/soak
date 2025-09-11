"""Data models for qualitative analysis pipelines."""

import hashlib
import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import anyio
import tiktoken
from box import Box
from struckdown import LLM, ChatterResult, LLMCredentials
from struckdown import chatter, chatter_async
from struckdown import get_embedding as get_embedding_
from struckdown.parsing import parse_syntax
from struckdown.return_type_models import ACTION_LOOKUP
from decouple import config as env_config
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateSyntaxError,
    meta,
)
from joblib import Memory
from pydantic import BaseModel, Field

from .document_utils import (
    extract_text,
    get_scrubber,
    unpack_zip_to_temp_paths_if_needed,
)

if TYPE_CHECKING:
    from .dag import QualitativeAnalysisPipeline


logger = logging.getLogger(__name__)

SOAK_MAX_RUNTIME = 60 * 60 * 3  # 3 hours


memory = Memory(Path(".embeddings"), verbose=0)


@memory.cache
def get_embedding(*args, **kwargs):
    return get_embedding_(*args, **kwargs)


def get_action_lookup():
    SOAK_ACTION_LOOKUP = dict(ACTION_LOOKUP.copy())
    SOAK_ACTION_LOOKUP.update(
        {
            "theme": Theme,
            "code": Code,
            "themes": Themes,
            "codes": CodeList,
        }
    )
    return SOAK_ACTION_LOOKUP


MAX_CONCURRENCY = env_config("MAX_CONCURRENCY", default=20, cast=int)
semaphore = anyio.Semaphore(MAX_CONCURRENCY)


# exception classes for backward compatibility
class CancelledRun(Exception):
    """Exception raised when a flow run is cancelled."""

    pass


class Cancelled(Exception):
    """Exception raised when a task is cancelled."""

    pass


class Code(BaseModel):
    # slug: str = Field(..., min_length=8, description="Unique identifier for the code")
    name: str = Field(..., min_length=1, description="A short name for the code.")
    description: str = Field(..., min_length=5, description="A description of the code.")
    quotes: List[str] = Field(
        ...,
        min_length=0,
        description="Example quotes from the text which illustrate the code. Choose the best examples.",
    )


class CodeList(BaseModel):
    codes: List[Code] = Field(..., min_length=0)

    def to_markdown(self):
        return "\n\n".join([f"- {i.name}: {i.description}\n{i.quotes}" for i in self.codes])


class Theme(BaseModel):
    name: str = Field(..., min_length=10)
    description: str = Field(..., min_length=10)
    # refer to codes by slug/identifier
    code_names: List[str] = Field(
        ...,
        min_length=0,
        max_length=10,
        description="List of the codes that are part of this theme. Identify them accurately by name",
    )


class Themes(BaseModel):
    themes: List[Theme] = Field(..., min_length=1)

    def to_markdown(self):
        return "\n- ".join([i.name for i in self.themes])


@dataclass
class Document:
    """Represents a source document (transcript)."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualitativeAnalysis(BaseModel):
    label: Optional[str] = None
    codes: Optional[List[Code]] = Field(default_factory=list)
    themes: Optional[List[Theme]] = Field(default_factory=list)
    narrative: Optional[str] = None
    quotes: Optional[Any] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)

    pipeline: Optional[str] = None

    def theme_text_for_comparison(self):
        return [i.name for i in self.themes]

    def name(self):
        return self.label or self.sha256()[:8]

    def sha256(self):
        return hashlib.sha256(json.dumps(self.model_dump()).encode()).hexdigest()[:8]

    def __str__(self):
        return f"Themes: {self.themes}\nCodes: {self.codes}"


class QualitativeAnalysisComparison(BaseModel):
    results: List["QualitativeAnalysis"]
    combinations: Any  # Dict[str, Tuple["QualitativeAnalysis", "QualitativeAnalysis"]]
    statistics: Dict[str, Dict]
    comparison_plots: Dict[str, Dict[str, Any]]  # eg. heatmaps.xxxyyy = List
    additional_plots: Dict[str, Any]
    config: dict

    def by_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a nested dict keyed by comparison key, with inner dict containing:
            - 'a', 'b' (the QualitativeAnalysis objects)
            - 'stats'
            - 'plots': { plot_type: plot_object, ... }
        """
        out = {}
        for key, (a, b) in self.combinations.items():
            out[key] = {"a": a, "b": b, "stats": self.statistics.get(key), "plots": {}}

        for plot_type in self.comparison_plots.keys():
            for k in out.keys():
                out[k]["plots"][plot_type] = self.comparison_plots[plot_type][k]

        return out


def get_default_llm_credentials():
    return LLMCredentials()


class DAGConfig(BaseModel):
    document_paths: Optional[List[str]] = []
    documents: List[str] = []
    model_name: str = "gpt-5-mini"
    temperature: float = 1.0
    chunk_size: int = 20000  # characters, so ~5k tokens or ~4k English words
    extra_context: Dict[str, Any] = {}
    llm_credentials: LLMCredentials = Field(default_factory=get_default_llm_credentials)
    scrub_pii: bool = False
    scrubber_model: str = "en_core_web_md"
    scrubber_salt: Optional[str] = Field(default="42", exclude=True)

    def get_model(self):
        return LLM(model_name=self.model_name, temperature=self.temperature)

    def load_documents(self) -> List[str]:
        if hasattr(self, "documents") and self.documents:
            logger.info("Using cached documents")
            return self.documents

        with unpack_zip_to_temp_paths_if_needed(self.document_paths) as dp_:
            self.documents = [extract_text(i) for i in dp_]

        if self.scrub_pii:
            logger.info("Scrubbing PII")
            if self.scrubber_salt == 42:
                logger.warning("Scrubber salt is default, consider setting to a random value")

            scrubber = get_scrubber(model=self.scrubber_model, salt=self.scrubber_salt)
            self.documents = [scrubber.clean(i) for i in self.documents]

        return self.documents


async def run_node(node):
    try:
        result = await node.run()
        logger.info(f"COMPLETED: {node.name}\n")
        return result
    except Exception as e:
        logger.error(f"Node {node.name} failed: {e}")
        raise e


def get_template_variables(template_string: str) -> Set[str]:
    """Extract all variables from a jinja template string.
    get_template_variables('{a} {{b}} c ')
    """
    env = Environment()
    ast = env.parse(template_string)
    return meta.find_undeclared_variables(ast)


def render_strict_template(template_str: str, context: dict) -> str:
    # try:
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(template_str)
    return template.render(**context)
    # except Exception as e:
    #     import pdb; pdb.set_trace()


@dataclass(frozen=True)
class Edge:
    from_node: str
    to_node: str


DAGNodeUnion = Annotated[
    Union[
        "Map",
        "Reduce",
        "Transform",
        "Batch",
        "Split",
        "TransformReduce",
        "VerifyQuotes",
    ],
    Field(discriminator="type"),
]


class DAG(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    default_context: Dict[str, Any] = {}
    default_config: Dict[str, Union[str, int, float]] = {}

    nodes: List["DAGNodeUnion"] = Field(default_factory=list)
    config: DAGConfig = DAGConfig()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # add defaults for config fields
        for k, v in self.default_config.items():
            if hasattr(self.config, k) and k not in self.config.model_fields_set:
                setattr(self.config, k, v)

    def progress(self):
        last_complete = self.nodes[0]
        return f"Last completed: {last_complete.name}"

    @property
    def edges(self) -> List["Edge"]:
        all_edges = []
        for node in self.nodes:
            for input_ref in node.inputs:
                if input_ref in [i.name for i in self.nodes]:
                    all_edges.append(Edge(from_node=input_ref, to_node=node.name))

        return all_edges

    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram of the DAG structure with shapes by node type."""
        lines = ["flowchart TD"]

        shape_map = {
            "Split": ("(", ")"),  # round edges
            "Map": ("[[", "]]"),  # standard rectangle
            "Reduce": ("{{", "}}"),  # hexagon
            "Transform": (">", "]"),  #
            "TransformReduce": (">", "]"),  #
            "VerifyQuotes": ("[[", "]]"),  #
            "Batch": ("[[", "]]"),  # subroutine shape
        }

        for node in self.nodes:
            le, ri = shape_map.get(node.type, ("[", "]"))  # fallback to rectangle
            label = f"{node.type}: {node.name}"
            lines.append(f"    {node.name}{le}{label}{ri}")

        for edge in self.edges:
            lines.append(f"    {edge.from_node} --> {edge.to_node}")

        lines.append("""classDef heavyDotted stroke-dasharray: 4 4, stroke-width: 2px;""")
        for node in self.nodes:
            if node.type == "TransformReduce":
                lines.append("""class all_themes heavyDotted;""")

        return "\n".join(lines)

    def get_execution_order(self) -> List[List[str]]:
        """Get the execution order as batches of nodes that can run in parallel."""
        remaining = set([i.name for i in self.nodes])
        execution_order = []

        while remaining:
            # Find nodes with no unprocessed dependencies
            ready = set()
            for node_name in remaining:
                deps = self.get_dependencies_for_node(node_name)
                if all(dep not in remaining for dep in deps):
                    ready.add(node_name)

            if not ready and remaining:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected in nodes: {remaining}")

            execution_order.append(list(ready))
            remaining -= ready

        return execution_order

    @property
    def nodes_dict(self):
        return {i.name: i for i in self.nodes}

    def cancel(self):
        if self.cancel_scope is not None:
            self.cancel_scope.cancel()
            logger.warning(f"DAG {self.name} cancelled")

    async def run(self):
        try:
            self.config.load_documents()
            if not self.config.llm_credentials:
                raise Exception("LLMCredentials must be set for DAG")
            for batch in self.get_execution_order():
                # use anyio structured concurrency - start all tasks in batch concurrently
                with anyio.fail_after(SOAK_MAX_RUNTIME):  # 2 hours, to cleanup if needed
                    async with anyio.create_task_group() as tg:
                        for name in batch:
                            tg.start_soon(run_node, self.nodes_dict[name])
                # all tasks in batch complete when task group exits
            return self, None
        except Exception as e:
            import traceback

            err = f"DAG execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(err)
            return self, str(e)

    def get_dependencies_for_node(self, node_name: str) -> Set[str]:
        """Get nodes that must complete before a node can run."""

        dependencies = set()

        # set[edge for edge in self.edges if edge.to_node == node_name]
        for edge in self.edges:
            if edge.to_node == node_name:
                dependencies.add(edge.from_node)

        return dependencies

    def add_node(self, node: "DAGNode"):
        # if self.nodes_dict.get(node.name):
        #     raise ValueError(f"Node '{node.name}' already exists in DAG")
        node.dag = self
        self.nodes.append(node)

    def get_required_context_variables(self):
        node_names = [i.name for i in self.nodes]
        tmplts = list(
            itertools.chain(*[get_template_variables(i.template) for i in self.nodes if i.template])
        )
        return set(tmplts).difference(node_names)

    def __str__(self):
        return f"DAG: {self.name}"

    def __repr__(self):
        return f"DAG: {self.name}"

    @property
    def context(self) -> Dict[str, Any]:
        """Backward compatibility: return node outputs as dict"""
        results = {v.name: v.output for v in self.nodes if v and v.output is not None}
        conf = self.config.extra_context.copy()
        conf.update(results)
        return conf


OutputUnion = Union[
    str,
    List[str],
    List[List[str]],
    ChatterResult,
    List[ChatterResult],
    List[List[ChatterResult]],
    # for top matches
    List[Dict[str, Union[str, List[Tuple[str, float]]]]],
]


class DAGNode(BaseModel):
    # this used for reserialization
    model_config = {"discriminator": "type"}
    type: str = Field(default_factory=lambda self: type(self).__name__, exclude=False)

    dag: Optional["DAG"] = Field(default=None, exclude=True)

    name: str
    inputs: Optional[List[str]] = []
    template_text: Optional[str] = None
    output: Optional[OutputUnion] = Field(default=None)

    def get_model(self):
        if self.model_name or self.temperature:
            m = LLM(model_name=self.model_name, temperature=self.temperature)
            return m
        return self.dag.config.get_model()

    def validate_template(self):
        try:
            Environment().parse(self.template_text)
            return True
        except TemplateSyntaxError as e:
            raise e
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self, items: List[Any] = None) -> List[Any]:
        logger.info(f"\n\nRunning `{self.name}` ({self.__class__.__name__})\n\n")

    @property
    def context(self) -> Dict[str, Any]:
        ctx = self.dag.default_context.copy()

        # merge in extra_context from config (includes persona, research_question, etc.)
        ctx.update(self.dag.config.extra_context)

        # if there are no inputs, assume we'll be using input documents
        if not self.inputs:
            self.inputs = ["documents"]

        if "documents" in self.inputs:
            ctx["documents"] = self.dag.config.load_documents()

        prev_nodes = {k: self.dag.nodes_dict.get(k) for k in self.inputs}
        prev_output = {k: v.output for k, v in prev_nodes.items() if v and v.output is not None}
        ctx.update(prev_output)

        return ctx

    @property
    def template(self) -> str:
        return self.template_text


class CompletionDAGNode(DAGNode):
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    async def run(self, items: List[Any] = None) -> List[Any]:
        await super().run()

    def output_keys(self) -> List[str]:
        """Return the list of output keys provided by this node."""
        try:
            sections = parse_syntax(self.template)
            keys = []
            for section in sections:
                keys.extend(section.keys())
            return keys or [self.name]
        except Exception as e:
            logger.warning(f"Failed to parse template for output keys: {e}")
            return ["input"]


class Split(DAGNode):
    type: Literal["Split"] = "Split"

    name: str = "chunks"
    template_text: str = "{{input}}"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences"] = "tokens"
    encoding_name: str = "cl100k_base"

    @property
    def template(self):
        return None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.inputs) > 1:
            raise ValueError("Split node can only have one input")

        if self.chunk_size < self.min_split:
            logger.warning(
                f"Chunk size must be larger than than min_split. Setting min_split to chunk_size // 2 = {self.chunk_size // 2}"
            )
            self.min_split = self.chunk_size // 2

    async def run(self) -> List[str]:
        import numpy as np

        await super().run()
        input_docs = self.context[self.inputs[0]]
        self.output = list(itertools.chain.from_iterable(map(self.split_document, input_docs)))
        lens = [len(self.tokenize(doc, method=self.split_unit)) for doc in self.output]
        logger.info(
            f"CREATED {len(self.output)} chunks; average length ({self.split_unit}): {np.mean(lens).round(1)}, max: {max(lens)}, min: {min(lens)}."
        )

        return self.output

    def split_document(self, doc: str) -> List[str]:
        if self.split_unit == "chars":
            return self._split_by_length(doc, len_fn=len, overlap=self.overlap)

        tokens = self.tokenize(doc, method=self.split_unit)
        spans = self._compute_spans(len(tokens), self.chunk_size, self.min_split, self.overlap)
        return [self._format_chunk(tokens[start:end]) for start, end in spans if end > start]

    def tokenize(self, doc: str, method: str) -> List[Union[int, str]]:
        if method == "tokens":
            return self.token_encoder.encode(doc)

        elif method == "sentences":
            import nltk

            return nltk.sent_tokenize(doc)
        elif method == "words":
            import nltk

            return nltk.word_tokenize(doc)
        else:
            raise ValueError(f"Unsupported tokenization method: {method}")

    def _compute_spans(
        self, n: int, chunk_size: int, min_split: int, overlap: int
    ) -> List[Tuple[int, int]]:
        if n <= chunk_size:
            return [(0, n)]

        n_chunks = max(1, math.ceil(n / chunk_size))
        target = max(min_split, math.ceil(n / n_chunks))
        spans = []
        start = 0
        for _ in range(n_chunks - 1):
            end = min(n, start + target)
            spans.append((start, end))
            start = max(0, end - overlap)
        spans.append((start, n))
        return spans

    def _format_chunk(self, chunk_tokens: List[Union[int, str]]) -> str:
        if not chunk_tokens:
            return ""
        if self.split_unit == "tokens":
            return self.token_encoder.decode(chunk_tokens).strip()
        elif self.split_unit in {"words", "sentences"}:
            return " ".join(chunk_tokens).strip()
        else:
            raise ValueError(f"Unexpected split_unit in format_chunk: {self.split_unit}")

    @property
    def token_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)


class ItemsNode(DAGNode):
    """Any note which applies to multiple items at once"""

    async def get_items(self) -> List[Dict[str, Any]]:
        """Resolve all inputs, then zip together, combining multiple inputs"""
        # resolve futures now (it's lazy to this point)
        input_data = {k: self.context[k] for k in self.inputs}

        lengths = {k: len(v) if isinstance(v, list) else 1 for k, v in input_data.items()}
        max_len = max(lengths.values())

        for k, v in input_data.items():
            if isinstance(v, list):
                if len(v) != max_len:
                    raise ValueError(
                        f"Length mismatch for input '{k}': expected {max_len}, got {len(v)}"
                    )
            else:
                input_data[k] = [v] * max_len

        zipped = list(zip(*[input_data[k] for k in self.inputs]))

        # make the first input available as {{input}} in any template
        items = []
        for values in zipped:
            item_dict = dict(zip(self.inputs, values))
            if self.inputs:
                item_dict["input"] = item_dict[self.inputs[0]]
            items.append(Box(item_dict))

        return items


async def default_map_task(template, context, model, credentials, **kwargs):
    """Default map task renders the Step template for each input item and calls the LLM."""

    rt = render_strict_template(template, context)

    # call chatter as async function within the main event loop
    result = await chatter_async(
        multipart_prompt=rt,
        context=context,
        model=model,
        credentials=credentials,
        action_lookup=get_action_lookup(),
        extra_kwargs=kwargs,
    )
    return result


# TODO implement scrubber as a node?
# class Scrub(ItemsNode):
#     type: Literal["Scrub"] = "Scrub"

#     def run(self) -> List[Any]:


class Map(ItemsNode, CompletionDAGNode):
    model_config = {
        "discriminator": "type",
    }

    type: Literal["Map"] = "Map"

    task: Callable = Field(default=default_map_task, exclude=True)
    template_text: str = None

    @property
    def template(self) -> str:
        return self.template_text

    def validate_template(self):
        try:
            parse_syntax(self.template_text)
            return True
        except Exception as e:
            logger.error(f"Template syntax error: {e}")
            return False

    async def run(self) -> List[Any]:
        # await super().run()

        input_data = self.context[self.inputs[0]] if self.inputs else None
        is_batch = isinstance(input_data, BatchList)

        # Flatten batch input if needed
        if is_batch:
            all_items = []
            batch_sizes = []
            for batch in input_data.batches:
                batch_items = [Box({"input": item}) for item in batch]
                all_items.extend(batch_items)
                batch_sizes.append(len(batch))
            items = all_items
            filtered_context = {
                k: v for k, v in self.context.items() if not isinstance(v, BatchList)
            }
        else:
            items = await self.get_items()
            filtered_context = self.context

        results = [None] * len(items)

        async with anyio.create_task_group() as tg:
            for idx, item in enumerate(items):

                async def run_and_store(index=idx, item=item):
                    async with semaphore:
                        results[index] = await self.task(
                            template=self.template,
                            context={**filtered_context, **item},
                            model=self.get_model(),
                            credentials=self.dag.config.llm_credentials,
                            max_tokens=self.max_tokens
                        )

                tg.start_soon(run_and_store)

        if is_batch:
            # Reconstruct BatchList structure
            reconstructed_batches = []
            result_idx = 0
            for batch_size in batch_sizes:
                batch_results = results[result_idx : result_idx + batch_size]
                reconstructed_batches.append(batch_results)
                result_idx += batch_size
            batch_list_result = BatchList(batches=reconstructed_batches)
            self.output = batch_list_result
            return batch_list_result
        else:
            self.output = results
            return results


class Transform(ItemsNode, CompletionDAGNode):
    type: Literal["Transform"] = "Transform"
    # TODO: allow for arbitrary python functions instead of chatter?
    # fn: Field(Callable[..., Iterable[Any]],  exclude=True) = None
    template_text: str = Field(default="{{input}} <prompt>: [[output]]")

    @property
    def template(self) -> str:
        return self.template_text

    async def run(self):
        items = await self.get_items()

        if not isinstance(items, str):
            assert len(items) == 1, "Transform nodes must have exactly one input item"

        rt = render_strict_template(self.template, {**self.context, **items[0]})

        # call chatter as async function within the main event loop
        self.output = await chatter_async(
            multipart_prompt=rt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=get_action_lookup(),
            extra_kwargs = {'max_tokens':self.max_tokens}   
        )
        return self.output


class Reduce(ItemsNode):
    type: Literal["Reduce"] = "Reduce"
    template_text: str = "{{input}}\n"

    @property
    def template(self) -> str:
        return self.template_text

    def get_items(self):
        if len(self.inputs) > 1:
            raise ValueError("Reduce nodes can only have one input")

        if self.inputs:
            input_data = self.dag.context[self.inputs[0]]
        else:
            input_data = self.dag.config.documents

        # if input is a BatchList, return it directly for special handling in run()
        if isinstance(input_data, BatchList):
            return input_data

        # otherwise, wrap individual items in the expected format
        nk = self.inputs and self.inputs[0] or "input_"
        items = [{"input": v, nk: v} for v in input_data]
        return items

    async def run(self, items=None) -> Any:
        await super().run()

        items = items or self.get_items()

        # if items is a BatchList, run on each batch
        if isinstance(items, BatchList):
            self.output = []
            for batch in items.batches:
                result = await self.run(items=batch)
                self.output.append(result)

            return self.output

        else:
            # handle both dictionaries and strings
            rendered = []
            for item in items:
                if isinstance(item, dict):
                    context = {**item}
                else:
                    # item is a string, wrap it for template processing
                    context = {"input": item}
                rendered.append(render_strict_template(self.template, context))
            self.output = "\n".join(rendered)
            return self.output


@dataclass
class BatchList(object):
    batches: List[Any]

    def __iter__(self):
        return iter(self.batches)


class Batch(ItemsNode):
    type: Literal["Batch"] = "Batch"
    # batch_fn: Optional[Callable] = None
    batch_size: int = 10

    async def run(self) -> List[List[Any]]:
        await super().run()

        batches_ = self.default_batch(await self.get_items(), self.batch_size)
        self.output = BatchList(batches=batches_)
        return self.output

    def default_batch(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Batch items into lists of size batch_size."""
        return list(itertools.batched(items, batch_size))


class QualitativeAnalysisPipeline(DAG):
    name: Optional[str] = None

    def to_html(self, template_path: Optional[str] = None) -> str:
        """Render the analysis as HTML using Jinja2 template from file.

        Args:
            template_path: Path to the HTML template file. If None, uses default template.

        Returns:
            Rendered HTML string.
        """
        if template_path is None:
            # Use default template in soak/templates directory
            template_dir = Path(__file__).parent / "templates"
            template_name = "qualitative_analysis.html"
        else:
            # Use provided template path
            template_path = Path(template_path)
            template_dir = template_path.parent
            template_name = template_path.name

        # Create Jinja2 environment and load template
        env = Environment(
            loader=FileSystemLoader(template_dir),
            extensions=["jinja_markdown.MarkdownExtension"],
        )

        template = env.get_template(template_name)

        # Render template with data
        dd = self.model_dump()
        dd['config']['documents'] =[]
        return template.render(pipeline=self, result=self.result(), detail=dd)

    def result(self):
        def safe_get_output(name):
            try:
                return self.nodes_dict.get(name).output.response
            except Exception:
                return None
            
        # import pdb; pdb.set_trace()
        return QualitativeAnalysis.model_validate(
            {
                "themes": safe_get_output("themes") or [],
                "codes":  safe_get_output("codes") or [],
                "narrative": safe_get_output("narrative"),
                "detail": self.model_dump_json(),
                "quotes": safe_get_output("quotes") or [],
            }
        )


class VerifyQuotes(DAGNode):
    type: Literal["VerifyQuotes"] = "VerifyQuotes"
    threshold: float = 0.6
    stats: Optional[Dict[str, Any]] = None

    async def run(self) -> List[Any]:
        await super().run()

        import nltk
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        alldocs = "\n\n".join(self.dag.config.documents)
        sentences = nltk.sent_tokenize(alldocs)
        real_quotes = sentences
        codes = self.context.get("codes", None)
        if not codes:
            raise Exception("VerifyQuotes must be run after node called `codes`")

        extr_quotes = list(itertools.chain(*[i.quotes for i in codes.response.codes]))

        # embed real and extracted quotes
        # import pdb; pdb.set_trace()
        try:
            real_emb = np.array(get_embedding(real_quotes))
            extr_emb = np.array(get_embedding(extr_quotes))
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()

        # calculate cosine similarity
        sims = cosine_similarity(extr_emb, real_emb)

        # find top matches and sort by similarity
        top_matches = []
        max_matches = 10
        for quote, row in zip(extr_quotes, sims):
            real_and_sim = list(zip(real_quotes, row))
            above_thresh = [(r, float(s)) for r, s in real_and_sim if s >= self.threshold]

            if above_thresh:
                matches = sorted(above_thresh, key=lambda x: -x[1])[:max_matches]
            else:
                top3_idx = row.argsort()[-3:]
                matches = sorted(
                    [(real_quotes[j], float(row[j])) for j in top3_idx],
                    key=lambda x: -x[1],
                )[:max_matches]

            top_matches.append(
                {
                    "quote": quote,
                    "matches": matches,
                }
            )

        # calulate stats on matches
        try:
            best_match_sims = [m["matches"][0][1] for m in top_matches if m["matches"]]
            average_sim = float(np.mean(best_match_sims)) if best_match_sims else None
            min_sim = float(np.min(best_match_sims)) if best_match_sims else None
            percentiles = (
                np.percentile(best_match_sims, [10, 90]) if best_match_sims else [None, None]
            )
            # Count no matches above threshold
            n_total = len(top_matches)
            n_below_thresh = sum(
                1 for m in top_matches if all(sim < self.threshold for _, sim in m["matches"])
            )
            pct_below_thresh = (n_below_thresh / n_total) * 100 if n_total else 0
            self.stats = {
                "average_best_sim": average_sim,
                "min_best_sim": min_sim,
                "10th_percentile_best_sim": (
                    float(percentiles[0]) if percentiles[0] is not None else None
                ),
                "90th_percentile_best_sim": (
                    float(percentiles[1]) if percentiles[1] is not None else None
                ),
                "n_no_match_above_threshold": n_below_thresh,
                "pct_no_match_above_threshold": round(pct_below_thresh, 2),
            }
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
            self.stats = {"error": str(e)}

        self.output = top_matches

        return top_matches


class TransformReduce(CompletionDAGNode):
    """
    Recursively reduce a list into a single item by transforming it with an LLM template.

    If inputs are ChatterResult with specific keys, then the TransformReduce must produce a ChatterResult with the same keys.

    """

    type: Literal["TransformReduce"] = "TransformReduce"

    chunk_size: int = 20000
    min_split: int = 500
    overlap: int = 0
    split_unit: Literal["chars", "tokens", "words", "sentences"] = "tokens"
    encoding_name: str = "cl100k_base"
    reduce_template: str = "{{input}}\n\n"
    template_text: str = "<text>\n{{input}}\n</text>\n\n-----\nSummarise the text: [[output]]"
    max_levels: int = 5

    reduction_tree: List[List[OutputUnion]] = Field(default_factory=list, exclude=False)

    @property
    def token_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)

    def tokenize(self, doc: str) -> List[Union[int, str]]:
        if self.split_unit == "tokens":
            return self.token_encoder.encode(doc)
        elif self.split_unit == "sentences":
            import nltk

            return nltk.sent_tokenize(doc)
        elif self.split_unit == "words":
            import nltk

            return nltk.word_tokenize(doc)
        else:
            raise ValueError(f"Unsupported tokenization method: {self.split_unit}")

    def _compute_spans(self, n: int) -> List[Tuple[int, int]]:
        if n <= self.chunk_size:
            return [(0, n)]
        n_chunks = max(1, math.ceil(n / self.chunk_size))
        target = max(self.min_split, math.ceil(n / n_chunks))
        spans = []
        start = 0
        for _ in range(n_chunks - 1):
            end = min(n, start + target)
            spans.append((start, end))
            start = max(0, end - self.overlap)
        spans.append((start, n))
        return spans

    def chunk_document(self, doc: str) -> List[str]:
        if self.split_unit == "chars":
            if len(doc) <= self.chunk_size:
                return [doc.strip()]
            spans = self._compute_spans(len(doc))
            return [doc[start:end].strip() for start, end in spans]
        tokens = self.tokenize(doc)
        spans = self._compute_spans(len(tokens))

        if self.split_unit == "tokens":
            return [self.token_encoder.decode(tokens[start:end]).strip() for start, end in spans]
        else:
            return [" ".join(tokens[start:end]).strip() for start, end in spans]

    def chunk_items(self, items: List[str]) -> List[str]:
        # import pdb; pdb.set_trace()
        joined = "\n".join(
            [
                render_strict_template(
                    self.reduce_template, {**self.context, **i.outputs, "input": i}
                )
                for i in items
            ]
        )
        return self.chunk_document(joined)

    def batch_by_units(self, items: List[str]) -> List[List[str]]:
        batches = []
        current_batch = []
        current_units = 0

        for item in items:
            if self.split_unit == "chars":
                item_len = len(item)
            elif self.split_unit == "tokens":
                item_len = len(self.token_encoder.encode(item))
            elif self.split_unit == "words":
                item_len = len(item.split())
            elif self.split_unit == "sentences":
                import nltk

                item_len = len(nltk.sent_tokenize(item))
            else:
                raise ValueError(f"Unknown split_unit: {self.split_unit}")

            # If adding this item would exceed batch size, flush
            if current_units + item_len > self.chunk_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_units = 0

            current_batch.append(item)
            current_units += item_len

        if current_batch:
            batches.append(current_batch)

        return batches

    async def run(self) -> str:
        await super().run()

        raw_items = self.context[self.inputs[0]]

        if isinstance(raw_items, str):
            raw_items = [raw_items]
        elif not isinstance(raw_items, list):
            raise ValueError("Input to TransformReduce must be a list of strings")

        # Initial chunking
        current = self.chunk_items(raw_items)

        self.reduction_tree = [current]

        level = 0
        nbatches = len(current)
        logger.info(f"Starting with {nbatches} batches")

        while len(current) > 1:
            level += 1
            logger.warning(f"TransformReduce level: {level}")
            if level > self.max_levels:
                raise RuntimeError("Exceeded max cascade depth")

            # Chunk input into batches
            batches = self.batch_by_units(current)
            if len(batches) > (nbatches and nbatches or 1e10):
                import pdb

                pdb.set_trace()
                raise Exception(
                    f"Batches {len(batches)} equals or exceeded original batch size {nbatches} so likely won't ever converge to 1 item. Try increasing the chunk_size or rewriting the prompt to be more concise."
                )
            nbatches = len(batches)

            logger.info(f"Cascade Reducing {len(batches)} batches")
            results: List[Any] = [None] * len(batches)

            async with anyio.create_task_group() as tg:
                for idx, batch in enumerate(batches):

                    async def run_and_store(index=idx, batch=batch):
                        prompt = render_strict_template(
                            self.template_text, {**self.context, "input": batch}
                        )
                        async with semaphore:
                            results[index] = await chatter_async(
                                multipart_prompt=prompt,
                                model=self.get_model(),
                                credentials=self.dag.config.llm_credentials,
                                action_lookup=get_action_lookup(),
                                extra_kwargs = {'max_tokens':self.max_tokens},
                            )

                    tg.start_soon(run_and_store)

            # Extract string results and re-chunk for next level
            current = self.chunk_items(results)
            self.reduction_tree.append(current)

        final_prompt = render_strict_template(
            self.template_text, {"input": current[0], **self.context}
        )
        final_response = await chatter_async(
            multipart_prompt=final_prompt,
            model=self.get_model(),
            credentials=self.dag.config.llm_credentials,
            action_lookup=get_action_lookup(),
            extra_kwargs = {'max_tokens':self.max_tokens}
        )

        self.output = final_response
        return final_response


# Resolve forward references after QualitativeAnalysisPipeline is defined

ItemsNode.model_rebuild(force=True)
DAGNode.model_rebuild(force=True)
DAG.model_rebuild(force=True)
QualitativeAnalysis.model_rebuild(force=True)
