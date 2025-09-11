"""Jinja2 template validation for pipeline templates."""

import logging
from typing import Dict, List, Set, Tuple

from jinja2 import Environment, meta
from jinja2.exceptions import TemplateSyntaxError

logger = logging.getLogger(__name__)


class TemplateValidationError(Exception):
    """Raised when template validation fails."""

    pass


class TemplateAnalyzer:
    """analyzes Jinja2 templates to extract variable dependencies."""

    def __init__(self):
        self.env = Environment()

    def get_template_variables(self, template_string: str) -> Set[str]:
        """extract all undeclared variables from a template string.

        Args:
            template_string: The template content to analyze

        Returns:
            Set of variable names that need to be provided in context

        Raises:
            TemplateValidationError: If template has syntax errors
        """
        try:
            ast = self.env.parse(template_string)
            return meta.find_undeclared_variables(ast)
        except TemplateSyntaxError as e:
            raise TemplateValidationError(f"Template syntax error: {e}")

    def analyze_template_dependencies(self, template_string: str) -> Dict[str, Set[str]]:
        """analyze all dependencies in a template.

        Args:
            template_string: The template content to analyze

        Returns:
            Dictionary with 'variables' and 'templates' keys containing
            sets of required variables and referenced templates
        """
        try:
            ast = self.env.parse(template_string or "")

            return {
                "variables": meta.find_undeclared_variables(ast),
                "templates": meta.find_referenced_templates(ast),
            }
        except TemplateSyntaxError as e:
            raise TemplateValidationError(f"Template syntax error: {e}")

    def validate_template_context(
        self, template_string: str, available_context: Set[str]
    ) -> Tuple[bool, List[str]]:
        """validate that all template variables are available in context.

        Args:
            template_string: The template content to validate
            available_context: Set of variable names available in context

        Returns:
            Tuple of (is_valid, list_of_missing_variables)
        """
        try:
            required_vars = self.get_template_variables(template_string)
            missing_vars = required_vars - available_context

            return len(missing_vars) == 0, list(missing_vars)
        except TemplateValidationError:
            return False, []

    def validate_chatter_template_syntax(self, template_string: str) -> Tuple[bool, str]:
        """validate that chatter template syntax is correct.

        Args:
            template_string: The template content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            from struckdown.parsing import parse_syntax

            # attempt to parse the template
            parse_syntax(template_string)
            return True, ""

        except Exception as e:
            error_msg = f"Chatter template syntax error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class PipelineTemplateValidator:
    """validates template dependencies across pipeline DAG nodes."""

    def __init__(self):
        self.analyzer = TemplateAnalyzer()

    def validate_node_template(
        self, template_path: str, available_context: Set[str], dag=None
    ) -> Tuple[bool, List[str]]:
        """validate a single node's template against available context.

        Args:
            template_path: Path to the template file
            available_context: Set of variable names available
            dag: Optional DAG instance for resolving template paths

        Returns:
            Tuple of (is_valid, list_of_missing_variables)
        """
        try:
            # resolve template path using DAG if provided
            if dag:
                full_path = dag.resolve_template_path(template_path)
            else:
                # fallback to old behavior
                full_path = f"soak/pipelines/{template_path}"

            with open(full_path, "r") as f:
                template_content = f.read()

            # first validate chatter template syntax
            syntax_valid, syntax_error = self.analyzer.validate_chatter_template_syntax(
                template_content
            )
            if not syntax_valid:
                return False, [syntax_error]

            # then validate jinja2 context
            return self.analyzer.validate_template_context(template_content, available_context)
        except FileNotFoundError:
            logger.error(f"Template file not found: {full_path}")
            return False, [f"Template file not found: {template_path}"]
        except Exception as e:
            logger.error(f"Error validating template {template_path}: {e}")
            return False, [f"Error reading template: {e}"]

    def get_template_variables(self, template_path: str, dag=None) -> Set[str]:
        """get all variables required by a template.

        Args:
            template_path: Path to the template file
            dag: Optional DAG instance for resolving template paths

        Returns:
            Set of variable names required by the template
        """
        try:
            # resolve template path using DAG if provided
            if dag:
                full_path = dag.resolve_template_path(template_path)
            else:
                # fallback to old behavior
                full_path = f"soak/pipelines/{template_path}"

            with open(full_path, "r") as f:
                template_content = f.read()

            return self.analyzer.get_template_variables(template_content)
        except FileNotFoundError:
            logger.error(f"Template file not found: {full_path}")
            return set()
        except Exception as e:
            logger.error(f"Error analyzing template {template_path}: {e}")
            return set()

    def suggest_fixes(
        self, template_path: str, missing_vars: List[str], available_context: Set[str]
    ) -> List[str]:
        """suggest fixes for missing template variables.

        Args:
            template_path: Path to the template with issues
            missing_vars: List of missing variable names
            available_context: Set of currently available variables

        Returns:
            List of human-readable suggestions
        """
        suggestions = []

        if not missing_vars:
            return suggestions

        suggestions.append(
            f"Template '{template_path}' is missing {len(missing_vars)} variable(s): {', '.join(missing_vars)}"
        )

        # suggest similar variables that might be typos
        for missing_var in missing_vars:
            similar_vars = [var for var in available_context if self._is_similar(missing_var, var)]
            if similar_vars:
                suggestions.append(
                    f"  - '{missing_var}' might be a typo. Similar available variables: {', '.join(similar_vars)}"
                )
            else:
                suggestions.append(
                    f"  - '{missing_var}' is not provided by any previous node or user input"
                )

        # suggest adding to user inputs
        user_input_candidates = [var for var in missing_vars if "." not in var and "_" not in var]
        if user_input_candidates:
            suggestions.append(
                f"Consider adding these to user inputs: {', '.join(user_input_candidates)}"
            )

        # suggest dependency fixes
        node_ref_candidates = [var for var in missing_vars if "." in var or "_" in var]
        if node_ref_candidates:
            suggestions.append(
                f"These look like node references - check dependency ordering: {', '.join(node_ref_candidates)}"
            )

        # show what context is available
        if available_context:
            context_list = sorted(available_context)
            suggestions.append(f"Available context variables: {', '.join(context_list[:10])}")
            if len(context_list) > 10:
                suggestions.append(f"... and {len(context_list) - 10} more")
        else:
            suggestions.append("No context variables are currently available")

        return suggestions

    def _is_similar(self, var1: str, var2: str, threshold: float = 0.6) -> bool:
        """check if two variable names are similar (simple similarity check)."""
        if len(var1) == 0 or len(var2) == 0:
            return False

        # simple similarity based on common characters
        common_chars = len(set(var1.lower()) & set(var2.lower()))
        max_len = max(len(var1), len(var2))

        return common_chars / max_len >= threshold
