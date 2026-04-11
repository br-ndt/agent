---
agents:
- image_generator
tags:
- image-gen
- critique
- limitations
- prompt-engineering
title: AI Image Generation Shortfalls Primer
updated_at: 1775704936.4245973
---

**Common AI Image Generation Limitations & Mitigation Strategies:**

1.  **Spatial Incoherence:** Models struggle with accurate depth, occlusion, and complex object interaction.
    *   **Mitigation:** Simplify layering. Avoid multiple interacting elements in initial prompts. Build composition iteratively.
2.  **Anatomical Distortion:** Frequent errors in human/animal anatomy (limbs, digits, proportions).
    *   **Mitigation:** Simplify figures or specify anatomical precision. Use reference images for complex poses.
3.  **Geometric Inconsistency:** Perspective lines, vanishing points, and structural integrity often fail.
    *   **Mitigation:** Explicitly describe architectural details and their relationships. Use terms like "converging lines," "consistent perspective."
4.  **Text & Detail Artifacts:** Generated text is usually nonsensical. Specific iconography or coherent text requires explicit instruction or post-processing.
    *   **Mitigation:** Avoid prompting for specific readable text unless using specialized models. Be highly specific for symbolic details.
5.  **Surface-Level Style Application:** Models apply styles as superficial textures rather than structural elements.
    *   **Mitigation:** Specify *how* a style affects form (e.g., "woodblock lines defining contours" vs. "woodblock style"). Guide structural execution.
6.  **Constraint Overload:** Too many interacting constraints (e.g., complex action + specific style + multiple figures + precise setting) lead to resolution failures.
    *   **Mitigation:** Isolate complex elements. Build composition by stabilizing base elements before adding interactions. Prioritize clarity over density in complex prompts.