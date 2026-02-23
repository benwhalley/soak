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