

Following code review of ~/dev/soaking and ~/dev/soakresearch


# Soaking

## alignment.py

- add docstring tests for snap_to_boundaries() with words and sentences as examples. Also - do a web search to check this isn't replicating something in nltk or similar package

- should snap_to_boundaries have a max_expansion arg? avoid extending more than 20 chars for word boundaries  and 200 for sentence boundaries. Set 20 and 200 as default arg max_expansion= on the function.


- are the lists of word and sentence boundaries standard? are they complete? are there predefined lists in existing packages like nltk?  are the sufficnetly multinational? 


- add docstring tests as examples to extract_context_window? can we get pytest to also run docstring tests?



- is HybridAlignment actually used anywhere? i.e. in any existing pipeline?


- trim_span_to_quote is described as backward compatible wrapper? could we remove it? is anything using it? or could we just reduce the indirection? e.g. get_alignment_strategy seems a bit superfluous if we don't actually use multiple alignment strategies? think this through a bit




## base.py

def compute_code_hash(name: str, description: str) -> str:
    """Compute deterministic hash for a code from its name and description."""
...

would this not be better as a method on the Code object? or a helper which just accepts a Code object? Otherwise it just seems to be a bit thin? Also, should think carefully about what fields constitute a unique Code. I think it is probably name and description, but potentially also it could include the Quote objects after post-processing and resoltion?
the argument here is that a Code with name=A, description=B and quotes C,D is not the same as a Code(A,B,[C,D,E,F]) or Code(A,B,[X,Y]) 

Perhaps Quote objects should have a similar hash function? Then the Code could hash those and include in the Code hash?

Finally, make sure we don't include the hash itself in the hash?!


---

SOAK_MAX_RUNTIME should be settable from an Env var, and in the soakresearch project it should be overriden by a constance config var in the DB


-------


> memory = Memory(Path(".embeddings"), verbose=0)
is this needed? shouldn't embeddings be cached within the struckown library? the hidden dir or cache location for embeddings should be controlled by struckdown not soaking?


-------

check that SOAK_MAX_CONCURRENCY can be overriden as constance config var in soakresearch project

-------


what is the point of  this?

def get_max_concurrency() -> int:
    """Get the current maximum concurrency value."""
    return _max_concurrency_value

why can't callers of this function just read the vaoue itself?
also - i can't see any callers wihtin soaking. is it called by soakresearch?  
or maybe it's just dead code? if so, clean up








tidy up [[code*:codes]] and [[theme*:themes]] syntax
maybe get rid CodeList and ThemeList?


use pandoc to replace markdown_to_typst and escape_typst?

tidy up tests in root dir




tidy up js like this - also check needed?
<script>
// ensure Enter key submits form (Safari workaround)
document.addEventListener('DOMContentLoaded', function() {
    var form = document.getElementById('login-form');
    if (form) {
        form.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.target.tagName !== 'BUTTON') {
                var submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    e.preventDefault();
                    submitBtn.click();
                }
            }
        });
    }
});