"""CLIP backbone used by the FDCA combiner (vendored). `clip.load(...)` builds
the model via `model.build_model`; text tokenization uses the external `clip`
package. Self-contained — no dependency on the training repo."""
