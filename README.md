# stark-brainfuck

Tutorial for designing and impementing a STARK-compatible VM, along with a fully functional Brainfuck instruction set architecture, virtual machine, prover, and verifier.

## Go to the Tutorial

The tutorial is available at [Github-Pages](https://aszepieniec.github.io/stark-brainfuck/).

## Running locally (the website, not the tutorial)

 1. Install ruby
 2. Install bundler
 3. Change directory to `docs/` and install Jekyll: `$> sudo bundle install`
 4. Run Jekyll: `$> bundle exec jekyll serve`
 5. Surf to [http://127.0.0.1:4000/](http://127.0.0.1:4000/)

## LaTeX and Github Pages

Github-Pages uses Kramdown as the markdown processor. Kramdown does not support LaTeX. Instead, there is a javascript header that loads MathJax, parses the page, and replaces LaTeX maths instructions with their proper formulae. Here is how to do it:

1. Open `_includes/head-custom.html` and paste the following code:
```javascript
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
    inlineMath: [ ['$', '$'], ['\\(', '\\)'] ],
    displayMath: [ ['$$', '$$'], ['\\[', '\\]'] ],
    processEscapes: true,
  }
});
MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
      alert("Math Processing Error: "+message[1]);
    });
MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
      alert("Math Processing Error: "+message[1]);
    });
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```
