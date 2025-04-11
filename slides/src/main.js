import Reveal from "reveal.js";
import Markdown from "reveal.js/plugin/markdown/markdown.esm";
import Highlight from "reveal.js/plugin/highlight/highlight.esm";
import Notes from "reveal.js/plugin/notes/notes.esm";
import Math from "reveal.js/plugin/math/math.esm";

import "reveal.js/dist/reveal.css";
import "reveal.js/dist/theme/black.css";
import "reveal.js/plugin/highlight/monokai.css";

import Slide from "./slides.md?raw";

let markdown = document.querySelector("#markdown");
markdown.innerHTML = Slide;

let deck = new Reveal({
  plugins: [Markdown, Highlight, Notes, Math.KaTeX],
});

deck.initialize();