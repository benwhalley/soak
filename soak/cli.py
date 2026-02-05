"""Backwards compatibility shim for CLI.

This module re-exports everything from the cli package for backwards compatibility.
The actual implementation is now in the soak/cli/ package.
"""

# re-export everything from the cli package
from .cli import (
    COMMANDS,
    PIPELINE_DIR,
    TEMPLATES_DIR,
    app,
    check_and_prompt_credentials,
    generate_all_html_outputs,
    generate_html_output,
    get_pdb_on_exception,
    get_soak_version,
    load_pipeline_json,
    main_with_default_command,
    resolve_analysis_path,
    resolve_template,
    set_pdb_on_exception,
    setup_logging,
    version_callback,
)

# for direct execution
if __name__ == "__main__":
    main_with_default_command()
