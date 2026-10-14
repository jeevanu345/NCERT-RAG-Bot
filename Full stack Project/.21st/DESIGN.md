# NCERT Tutor design context

A light, focused study workspace for Class 10 students. The user requested a simple, aesthetic, intuitive redesign and NVIDIA NIM BYOK.

- White study canvas, sage sidebar, forest-green actions, muted pastel study-mode icons.
- Avenir Next/Segoe UI system text with Iowan Old Style/Georgia display headings. No runtime font downloads.
- Clear navigation: Study space, Textbook library, Saved answers.
- Button and Input reuse the installed shadcn primitives; dialogs use Radix for focus management.
- A single composer contains textbook selection, four study modes, attachment, and send/stop.
- Starter prompts fill the composer for review before sending.
- Source passages are expandable; PDF page numbers are explicit.
- Mobile navigation uses a modal drawer. Reduced motion, focus outlines, and error recovery are required.

The 21st reference search included [Sidebar Light](https://21st.dev/@inference-sh/components/sidebar-light) and [Conversation List](https://21st.dev/@arihantcodes_1f7b8c4d/components/conversation-list). No external component source was installed. The hosted source review was rejected by automatic approval review, so validation uses local checks and browser tests.

Actual tokens and durable implementation choices are recorded in `design.json`; `src/app/globals.css` remains the source of truth.
