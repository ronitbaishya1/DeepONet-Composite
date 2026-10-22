# B9A11W: weak-form equilibrium and surface-readiness research

This directory captures fixed-interface diagnostic studies, **not** a fully coupled or experimentally validated digital twin.

- **W0:** Orthotropic C3D8R candidate force audit on 24 previously inspected Abaqus reference cases, original B5 and sensitivity-trained B5. Force law is B9A10L with empirical B9A10K hourglass correction; it is *not* exact Abaqus physics or approved for production inverse/coupling.
- **W1:** Geometrically eligible exterior faces for future strain gauges. Candidate face normals do not prove absence of contact or validity of surface strain measurements.
- **W2:** Guarded test-time refinement with fixed FEM interface state q. The `full_same_settings01` batch uses 12 cases/region, six modes, one step and 1e-5 mm RMS correction cap. Average candidate force residuals improve; individual FEM displacement and interface force components can worsen.

Large ODB/NPZ datasets and model checkpoints were not committed. Existing B5 weights and the 65V state are unchanged.
