uv run soak -v run zs data/cfs/\*.txt -o cfs1 --model-name="litellm/gpt-4.1-mini"  -t simple -t pipeline

uv run soak -v run zs data/cfs/\*.txt -o cfs2 --model-name="litellm/gpt-4.1"  -t simple -t pipeline

uv run soak -v compare cfs1.json cfs2.json --output=cfs_comparison.html 
