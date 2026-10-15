# NCERT Tutor design context

A quiet, polished study workspace and editorial landing page. The user supplied a green open-book/person logo and requested dark mode, selectable palettes/fonts, detailed animation, and Google/Microsoft/Apple account flows.

- Native vector brand mark: dark teal outer book, green/sage pages, head and three rays; bold NCERT and light tutor wordmark.
- Light, dark, and system modes; Forest, Ocean, Iris, Rose, Amber, and Slate accents. Semantic surfaces and states adapt throughout the interface.
- Self-hosted Manrope, Inter, Nunito Sans, or system body fonts; Lora, matching body font, or Georgia display headings. No third-party font service.
- Settings uses keyboard-accessible Radix tabs, a palette grid, font selects, and a live appearance preview. Choices persist on the device.
- The landing page explains the real product with interactive study examples, three setup steps, feature illustrations, data-handling information, FAQs, and calls to action. Examples are labelled and never make API calls.
- Short entrance and scroll transitions respect reduced motion. A decorative source-chip movement ends after four seconds. Content remains visible without JavaScript.
- Sign-in, sign-out, provider-error, and privacy pages share the same appearance. Account controls appear in the landing header and workspace.
- Study navigation, source disclosures, textbook library, saved answers, and focused composer remain familiar. Mobile navigation uses a modal drawer.

21st searches grounded the layout in light sidebars, conversation lists, editorial heroes, and feature grids. No external component source was installed. A previous hosted source review was rejected by automatic approval review; local browser, responsive, and accessibility checks are used instead.

`src/app/appearance.css` is the source of truth for appearance tokens; `src/app/globals.css` and `src/app/landing.css` define the component layouts. Durable choices are recorded in `design.json`.
