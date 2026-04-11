---
title: 3D web learnings
agents:
- coder
- architect
- researcher
- ops
tags:
- web-dev
- 3d
- 3js
- web-app
updated_at: 1775704936.4245973
---

## Three.js Core Patterns

1. **Always dispose of resources.** GPU memory leaks are the #1 production killer. Call `.dispose()` on geometries, materials, textures, and render targets when removing objects. A scene that runs fine for 5 minutes can crash a tab after 30 if you skip this.
    * Use `renderer.info.memory` to monitor geometry/texture counts during development.
    * When removing a mesh: `mesh.geometry.dispose(); mesh.material.dispose(); mesh.material.map?.dispose(); scene.remove(mesh);`

2. **Reuse geometries and materials.** Creating a new `BoxGeometry` and `MeshStandardMaterial` per instance is catastrophic at scale. Use `InstancedMesh` for repeated objects (trees, particles, tiles) — one draw call for thousands of instances.
    * `InstancedMesh` + a single material + `setMatrixAt()` per instance is the standard pattern.
    * For varied colors, use `instanceColor` attribute or vertex colors rather than separate materials.

3. **The render loop is sacred.** Never do heavy computation in `requestAnimationFrame`. Offload physics, pathfinding, and procedural generation to Web Workers. Use `SharedArrayBuffer` or message-passing to sync positions back.
    * Target 16ms per frame (60fps). If your update logic takes 8ms, you have 8ms for rendering. Monitor with `renderer.info.render.calls` and `performance.now()`.

4. **Use BufferGeometry, never Geometry.** `Geometry` was removed in r125+. All custom geometry should use `BufferGeometry` with typed arrays (`Float32Array` for positions, `Uint16Array` for indices).

## Performance & Optimization

5. **Draw calls are the bottleneck, not polygon count.** 100 objects with 1000 triangles each is far worse than 1 merged object with 100,000 triangles. Merge static geometry with `BufferGeometryUtils.mergeGeometries()`.
    * Target under 100-200 draw calls for mobile. Desktop can handle 500-1000.
    * `renderer.info.render.calls` tells you exactly where you stand.

6. **LOD (Level of Detail) for anything with distance.** Use `THREE.LOD` to swap high-poly models for low-poly versions at distance. For terrain/landscapes, this is non-negotiable.

7. **Frustum culling is free — use it.** Three.js culls by default (`object.frustumCulled = true`), but only at the object level. Large objects that are partially visible won't be culled. Break large meshes into chunks.

8. **Texture atlases over many small textures.** Each texture is a separate GPU upload and potentially a separate draw call. Pack related textures into atlases and use UV offsets.
    * Use power-of-two dimensions (512x512, 1024x1024, 2048x2048) — non-POT textures force fallback behavior on some GPUs.
    * Compress with KTX2/Basis Universal for massive size wins (70-80% smaller than PNG). `KTX2Loader` + `BasisTextureLoader` in Three.js.

9. **Shadow maps are expensive.** Use them sparingly. Bake shadows for static scenes. For dynamic shadows, use small shadow map sizes (512-1024), limit shadow-casting lights to 1-2, and use `shadow.camera` to tightly bound the shadow frustum.

10. **Post-processing stacks add up.** Each pass is a full-screen quad render. Combine passes where possible. `EffectComposer` with `RenderPass` + `UnrealBloomPass` + `SMAAPass` is already 3 full-screen draws.

## Camera & Controls

11. **OrbitControls for inspection, PointerLockControls for FPS, MapControls for top-down.** Don't build custom camera controls unless you need something truly novel — the built-in controls handle edge cases (gimbal lock, damping, zoom limits) that take weeks to get right.
    * Always set `controls.minDistance` and `controls.maxDistance` to prevent users from zooming into or through objects.
    * Enable `controls.enableDamping = true` with `controls.dampingFactor = 0.05` for smooth feel — but you must call `controls.update()` in the render loop.

12. **Camera near/far ratio matters.** A near plane of 0.001 and far plane of 100000 gives terrible z-fighting. Keep the ratio under 10000:1. For large scenes, use logarithmic depth buffer: `new WebGLRenderer({ logarithmicDepthBuffer: true })`.

## Loading & Assets

13. **GLTF is the standard.** Use `.glb` (binary GLTF) for everything — models, animations, entire scenes. It's compressed, fast to parse, and supports PBR materials natively.
    * `GLTFLoader` + `DRACOLoader` for mesh compression (30-60% smaller).
    * `gltf-transform` CLI for optimizing/compressing existing models outside the browser.

14. **Lazy-load assets.** Don't block the initial render on loading 50MB of models. Show a loading screen or progressively load assets. `THREE.LoadingManager` tracks overall progress.
    * For large worlds, load chunks based on camera position (spatial partitioning).

15. **Preload textures before creating materials.** `TextureLoader.load()` is async. If you create a material with an unloaded texture, you'll get a flash of default material. Load textures first, then build materials in the callback.

## Lighting

16. **Three lights max for real-time.** `AmbientLight` (fill) + `DirectionalLight` (sun/key) + `HemisphereLight` (sky/ground gradient) covers 90% of outdoor scenes. Each additional light multiplies shader complexity.
    * `PointLight` and `SpotLight` are much more expensive than `DirectionalLight`. Use them sparingly and set `light.distance` to limit their range.

17. **Bake lighting for static scenes.** For architectural visualization or environments that don't change, bake lightmaps in Blender and apply them as `lightMap` on materials. Zero runtime cost.

18. **Environment maps for reflections.** Use `PMREMGenerator` to create prefiltered environment maps from HDR images. Apply to scene via `scene.environment`. Far cheaper than real-time reflections.

## Common Pitfalls

19. **Y is up in Three.js, Z is up in Blender.** Models exported from Blender will be rotated 90 degrees. Either fix in Blender before export (apply transforms) or rotate the root object.

20. **Coordinate scale matters.** Three.js has no unit system — 1 unit is whatever you decide. But physics engines (cannon-es, rapier) assume meters. Pick meters as your unit and stick with it. A character should be ~1.7 units tall, not 170.

21. **Transparent objects must be sorted.** Three.js sorts transparent objects by distance, but it's per-object, not per-pixel. Overlapping transparent objects will render incorrectly. Solutions: avoid transparency, use alpha testing (`alphaTest: 0.5`), or use OIT techniques.

22. **Double-sided materials cost double.** `material.side = THREE.DoubleSide` renders both faces. Use it only where needed (thin surfaces like leaves, paper). For solid objects, fix normals instead.

23. **Mobile GPUs are a different world.** Half the VRAM, no compute shaders, limited texture units. Test on actual devices. Use `renderer.capabilities` to detect limits and degrade gracefully. Halve texture resolutions and shadow map sizes as a starting point.

## Architecture Patterns

24. **ECS (Entity Component System) scales better than scene graph.** For games or complex simulations, Three.js's scene graph is a rendering structure, not an architecture. Use an ECS library (bitecs, miniplex) for game logic and sync positions to Three.js meshes.

25. **Separate simulation from rendering.** Run physics/game logic at a fixed timestep (e.g., 60Hz) and interpolate for rendering. This prevents physics from breaking at high/low framerates.
    * Fixed timestep pattern: accumulate `dt`, step simulation in fixed increments, interpolate visual positions between last two physics states.

26. **State management for UI overlays.** Use your framework's state (React/Vue/Svelte) for HUD, menus, and UI. Don't try to render UI in WebGL. React Three Fiber's `<Html>` component bridges the two worlds cleanly.

## React Three Fiber Specifics

27. **R3F is declarative Three.js.** Components map 1:1 to Three.js objects. `<mesh>` is a `THREE.Mesh`, `<boxGeometry>` is `THREE.BoxGeometry`. Props map to constructor args and properties.
    * `useFrame((state, delta) => { ... })` is the render loop hook. Keep it light.
    * `useThree()` gives access to the renderer, camera, scene, and gl context.

28. **Drei is the utility belt.** Don't rebuild common patterns — `@react-three/drei` has `OrbitControls`, `Environment`, `Text`, `Html`, `useGLTF`, `useTexture`, `Instances`, `Float`, `Billboard`, and dozens more.

29. **Suspense for loading.** Wrap your scene in `<Suspense fallback={<LoadingScreen />}>`. Asset loaders in R3F (`useGLTF`, `useTexture`) suspend automatically.

30. **Avoid re-renders.** React re-renders are expensive when they trigger Three.js object recreation. Memoize components, use `useMemo` for geometries/materials, and keep rapidly-changing state (positions, rotations) in refs, not React state.

## Networking & Multiplayer

31. **WebSockets for real-time, not WebRTC.** WebRTC data channels have lower latency but the connection setup is complex. For most multiplayer web games, WebSocket with a relay server is simpler and sufficient.
    * Send position/rotation updates at 10-20Hz, interpolate on the client. Don't send every frame.
    * Use binary protocols (ArrayBuffer) not JSON for position data. 12 bytes (3x float32) vs ~50 bytes of JSON per position update.

32. **Client-side prediction + server reconciliation.** For responsive multiplayer, apply inputs locally immediately, send to server, and correct when the server response arrives. This is hard to get right — use a library if possible.

## Deployment

33. **Bundle size matters.** Three.js core is ~600KB minified. Import only what you use: `import { Scene, Mesh } from 'three'` not `import * as THREE from 'three'`. Tree-shaking works if you use ES module imports.

34. **Compress everything.** Brotli for JS/WASM, KTX2/Basis for textures, Draco for meshes. A 50MB uncompressed scene can be 5MB compressed.

35. **Web Workers for heavy lifting.** Mesh generation, texture processing, physics simulation — anything that takes more than a few ms should be off the main thread. `Comlink` simplifies Worker communication.

36. **Test on potato hardware.** Your MacBook Pro is not your users' machine. Test on a 3-year-old Android phone and an integrated-GPU laptop. Use `stats.js` or the browser's performance panel to catch frame drops early.
