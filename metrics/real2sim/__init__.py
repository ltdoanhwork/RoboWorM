"""Real-to-sim reconstruction: build the scene that is being measured, from the video itself.

The cheap tier segments; ``force_feasibility`` applies Newton to a *proxy* (a box for the
object, two boxes for the fingers) because that needs no per-object asset. This module
removes that approximation: a single-image-to-3D model turns the object's own masked crop
into a watertight mesh, so contact geometry and inertia come from the observed shape rather
than from a bounding box.

What this fixes and what it does not
------------------------------------
Reconstruction gives *shape*, in a normalised unit cube. It does not give **metres** -- the
generator has no way to know how large the object is -- and metric scale is what the force
tier is most sensitive to (grip demand is proportional to it). So the scale question is
solved separately, in ``probe_static_scale``, and reconstruction is layered on top of it.
"""
