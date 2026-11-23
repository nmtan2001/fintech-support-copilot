---
name: FablePay Support Assistant
description: A single-conversation interface for fictional finance support.
colors:
  ink: "#202523"
  muted: "#59645e"
  accent: "#24624b"
  line: "#e2e7e3"
  canvas: "#ffffff"
  sidebar: "#f5f7f5"
  button-border: "#ccd5ce"
  button-hover: "#eaf1ec"
  button-hover-border: "#8fae9c"
  button-hover-ink: "#184a35"
  disabled-ink: "#727b76"
  disabled-surface: "#eff2ef"
  user-surface: "#edf3ee"
  input-border: "#bccbc0"
  selection-surface: "#cee5d7"
  selection-ink: "#163b2d"
typography:
  body:
    fontFamily: '"Segoe UI", system-ui, sans-serif'
    lineHeight: 1.65
  headline:
    fontSize: "2rem"
    fontWeight: 600
    letterSpacing: "-.025em"
  title:
    fontSize: "1.05rem"
    fontWeight: 600
  label:
    fontSize: ".8rem"
  message-caption:
    fontSize: ".75rem"
    lineHeight: 1.5
rounded:
  button: "10px"
  conversation: "16px"
spacing:
  quarter: ".25rem"
  half: ".5rem"
  three-quarter: ".75rem"
  one: "1rem"
  two: "2rem"
components:
  button:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    rounded: "{rounded.button}"
  button-hover:
    backgroundColor: "{colors.button-hover}"
    textColor: "{colors.button-hover-ink}"
  button-disabled:
    backgroundColor: "{colors.disabled-surface}"
    textColor: "{colors.disabled-ink}"
  composer:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    rounded: "{rounded.conversation}"
  user-message:
    backgroundColor: "{colors.user-surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.conversation}"
    padding: "1rem 1.2rem"
---

# Design System: FablePay Support Assistant

## Overview

**Creative North Star: "Conversation first"**

This describes the shipped interface rather than proposing a new brand identity. FablePay is a fictional finance support demo with one active conversation. A white transcript, pale sidebar, restrained green accent, and compact support header keep the question and answer central.

Useful task prompts introduce the empty state. Sources, record details, review details, and diagnostics are progressively disclosed. Demo ownership and account limitations remain visible in the sidebar. This documentation records the existing presentation; it does not change backend or API semantics.

**Key Characteristics:**

- One active conversation with a bottom composer.
- Quiet neutral surfaces and dark readable text.
- Three task prompts before the first exchange.
- Collapsed supporting details and sidebar diagnostics.

## Colors

The frontmatter records the shipped palette. Color names describe observed roles.

### Primary

- **Support green** (`accent`): focus outlines, composer caret, submit control, and Streamlit's primary theme color.
- **Hover greens** (`button-hover`, `button-hover-border`, `button-hover-ink`): pale fill, stronger border, and dark button text.

### Neutral

- **Charcoal ink** (`ink`) and **muted gray-green** (`muted`): primary and supporting text. Captions and placeholders retain full opacity.
- **White canvas** (`canvas`): application, header, bottom composer area, and resting buttons.
- **Sidebar mist** (`sidebar`): separates demo controls from the transcript.
- **Conversation tint** (`user-surface`): distinguishes user messages.
- **Quiet borders** (`line`, `button-border`, `input-border`): separators and control boundaries.
- **Disabled neutrals** (`disabled-ink`, `disabled-surface`): unavailable prompt and record actions.
- **Selection colors** (`selection-surface`, `selection-ink`): browser text selection.

## Typography

Segoe UI uses system and sans-serif fallbacks. Base body size and unoverridden control metrics come from Streamlit; no separate display font is shipped.

- **Headline:** the welcome question uses the headline role, reducing to 1.75rem on small screens.
- **Title:** the compact support header uses the title role. Sidebar product text is 1.35rem; section headings and starter prompts are .875rem.
- **Body:** transcript content uses the body line height. Assistant content has a 75ch maximum; welcome supporting text has a 70ch maximum with balanced wrapping.
- **Captions:** message labels use the message-caption role; captions remain fully opaque. Header metadata uses the label role.

## Layout

Main content and the bottom composer share an 860px maximum container width. Desktop main padding is 3.5rem 2rem 8rem; bottom padding is 1rem 2rem 1.5rem. The sidebar owns customer selection, sample records, New conversation, diagnostics, and the review queue. There is no conversation history navigation.

The empty state begins at `min(16vh, 9rem)` and presents three prompts across columns. User messages align right with an 85% maximum width. Assistant messages retain the open transcript surface. Messages have 1rem vertical padding and a .8rem gap.

At 640px and below, main padding becomes 3rem 1rem 7rem and bottom padding becomes .75rem 1rem 1rem. Header metadata stacks below the title, welcome spacing becomes 3rem, and prompt buttons stack in full-width columns. Sidebar opening and closing follows Streamlit behavior.

## Elevation & Depth

Tonal surfaces and thin borders provide most separation. The composer alone receives the application's soft resting shadow: `0 4px 16px rgba(26, 45, 34, .06)`. Messages and ordinary buttons receive no added application shadow.

## Shapes

Buttons use the smaller button radius; the composer and user bubbles use the larger conversation radius. Straight divider lines separate the sidebar and compact header. Assistant replies remain unboxed. There is no separate card collection or navigation chip system.

## Components

### Buttons

White outlined buttons carry sidebar actions and starter prompts. Prompts have a 3.5rem minimum height. Hover uses the green fill, border, and ink roles. Buttons use a 2px accent focus outline with a 3px offset; select controls receive the same treatment on focus within. Disabled controls use the disabled neutral pair and divider border.

Background and border transitions last 150ms with `ease-out`. Reduced-motion preferences remove these transitions. No custom pressed animation is defined.

### Composer

The white composer uses the input border, conversation radius, and sole ambient shadow. Focus within adds a 2px accent outline with a 2px offset. The placeholder is “Message FablePay support”; input is limited to 2,000 characters. Sending displays “Checking your question…” while awaiting a response.

### Conversation messages

User bubbles retain a “You” caption and hide the user avatar. Assistant replies use a support-agent avatar and “FablePay assistant” caption. Responses render Markdown and numbered citation references.

Sources and Record details are closed expanders by default. A local review handoff adds an explicit demo-only caption and closed Review request details expander. Unverified generation displays an explanatory information message; request errors use error messages.

### Empty state

“How can I help?” introduces Check a card payment, Track a transfer, and Report a lost card. Each sends a question using an available record belonging to the selected fictional customer. Missing matching records disable their actions. The empty state disappears after the first completed exchange.

### Sidebar controls

New conversation clears the displayed exchange and resets the selected session. Customer selection locks once a conversation exists; a caption explains how to switch. A record picker and Ask about this record send a status question. Conversation details and Review queue are closed expanders when their associated state exists. Demo limitations explicitly state that the assistant cannot move money or change an account.

## Do's and Don'ts

### Do:

- **Do** keep one active transcript and a bottom composer.
- **Do** preserve full-opacity captions and the 75ch assistant maximum.
- **Do** keep evidence and diagnostics available through collapsed details.
- **Do** retain visible fictional-demo context and account limitations.
- **Do** preserve keyboard focus, disabled states, and reduced-motion behavior.

### Don't:

- **Don't** introduce multiple-conversation navigation into this experience.
- **Don't** expand raw records or technical diagnostics into the default answer view.
- **Don't** imply that local review requests contact a real bank.
